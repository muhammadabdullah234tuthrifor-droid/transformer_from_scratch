import argparse
import contextlib
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import IGNORE_INDEX, build_dataloaders
from generate import Seq2SeqLM


# ----------------------------- config -----------------------------
@dataclass
class TrainConfig:
    data_dir: str = "processed_linux_cmd"
    out_dir: str = "runs/linux_cmd_v1"
    # model
    inr_dim: int = 256
    lnr_dim: int = 768
    nm_hds: int = 4
    nm_lyrs: int = 3
    dropout: float = 0.1
    tie_embeddings: bool = True
    # optimisation
    epochs: int = 40
    batch_size: int = 64
    lr: float = 5e-4
    min_lr_ratio: float = 0.05
    warmup_steps: int = 200
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    amp: str = "none"              # "none" | "bf16" | "fp16"
    # evaluation / checkpoints
    monitor: str = "val_loss"      # "val_loss" | "val_em"
    patience: int = 8
    max_gen_len: int = 96
    show_samples_every: int = 5
    log_every: int = 50
    num_workers: int = 0
    seed: int = 42


# ----------------------------- helpers -----------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def get_amp_dtype(cfg, device):
    if device.type != "cuda" or cfg.amp == "none":
        return None
    return torch.bfloat16 if cfg.amp == "bf16" else torch.float16


def autocast_ctx(amp_dtype, device):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def build_model(cfg, meta):
    return Seq2SeqLM(
        vocab_size=meta["vocab_size"], mx_sq_ln=meta["mx_sq_ln"],
        inr_dim=cfg.inr_dim, lnr_dim=cfg.lnr_dim, nm_hds=cfg.nm_hds,
        nm_lyrs=cfg.nm_lyrs, pad_tok_id=meta["pad_tok_id"],
        dropout=cfg.dropout, tie_embeddings=cfg.tie_embeddings,
    )


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "embedding" in name else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=1e-9)


def build_scheduler(optimizer, cfg, total_steps):
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cut_at(ids, stop_id):
    return ids[:ids.index(stop_id)] if stop_id in ids else ids


def first_word(text):
    parts = text.split()
    return parts[0] if parts else ""


# ----------------------------- train / eval -----------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg,
                    device, amp_dtype, epoch, step):
    model.train()
    sum_loss = torch.zeros((), device=device)
    n_batches = 0
    for batch in loader:
        b = to_device(batch, device)
        with autocast_ctx(amp_dtype, device):
            logits = model(b["src_ids"], b["tgt_ids"])
        loss = F.cross_entropy(                      # computed in fp32 on purpose
            logits.float().reshape(-1, model.vocab_size),
            b["labels"].reshape(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=cfg.label_smoothing,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        n_batches += 1
        sum_loss += loss.detach()
        if step % cfg.log_every == 0:
            print(f"  ep {epoch:02d} step {step:5d} | loss {loss.item():.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | gnorm {gnorm:.2f}")
    return (sum_loss / max(1, n_batches)).item(), step


@torch.no_grad()
def evaluate_loss(model, loader, device, amp_dtype):
    model.eval()
    total, tokens = 0.0, 0
    for batch in loader:
        b = to_device(batch, device)
        with autocast_ctx(amp_dtype, device):
            logits = model(b["src_ids"], b["tgt_ids"])
        total += F.cross_entropy(
            logits.float().reshape(-1, model.vocab_size), b["labels"].reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum",
        ).item()
        tokens += (b["labels"] != IGNORE_INDEX).sum().item()
    nll = total / tokens
    return nll, math.exp(min(nll, 20))


@torch.no_grad()
def evaluate_generation(model, loader, tok, meta, cfg, device, amp_dtype, n_samples=3):
    model.eval()
    bos, eos = meta["bos_tok_id"], meta["eos_tok_id"]
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True).strip()
    n = exact = tool = 0
    samples = []
    for batch in loader:
        b = to_device(batch, device)
        with autocast_ctx(amp_dtype, device):
            pred = model.generate(b["src_ids"], bos, eos, cfg.max_gen_len)
        for src_row, lbl_row, pred_row in zip(
            batch["src_ids"].tolist(), batch["labels"].tolist(), pred.tolist()
        ):
            ref_ids = [t for t in lbl_row if t != IGNORE_INDEX and t != eos]
            pred_ids = cut_at(pred_row, eos)
            ref_txt, pred_txt = decode(ref_ids), decode(pred_ids)
            n += 1
            exact += int(pred_ids == ref_ids)
            tool += int(first_word(pred_txt) == first_word(ref_txt))
            if len(samples) < n_samples:
                samples.append((decode(cut_at(src_row, eos)), ref_txt, pred_txt))
    return {"em": exact / n, "tool_acc": tool / n, "n": n}, samples


def print_samples(samples):
    for src, ref, pred in samples:
        print(f"    NL  : {src}\n    REF : {ref}\n    PRED: {pred}\n")


# ----------------------------- sanity checks -----------------------------
@torch.no_grad()
def check_masks_and_init(model, batch, meta, device):
    model.eval()
    b = to_device(batch, device)
    src, tgt, lbl = b["src_ids"], b["tgt_ids"], b["labels"]
    pad, V = meta["pad_tok_id"], meta["vocab_size"]
    base = model(src, tgt)

    loss = F.cross_entropy(base.reshape(-1, V), lbl.reshape(-1), ignore_index=IGNORE_INDEX)
    print(f"[1] initial loss {loss.item():.3f}  (ln V = {math.log(V):.3f}) "
          f"-> should be within ~1 of ln V")

    extra = min(7, meta["mx_sq_ln"] - max(src.size(1), tgt.size(1)))
    if extra > 0:
        out2 = model(F.pad(src, (0, extra), value=pad), F.pad(tgt, (0, extra), value=pad))
        err = (base - out2[:, : tgt.size(1)]).abs().max().item()
        print(f"[2] padding invariance max|diff| = {err:.2e}  "
              f"{'PASS' if err < 1e-3 else 'FAIL (a mask is leaking padding)'}")
    else:
        print("[2] skipped (batch already at mx_sq_ln)")

    if tgt.size(1) >= 2:
        tgt_mod = tgt.clone()
        tgt_mod[:, -1] = (tgt_mod[:, -1] + 1) % V
        out3 = model(src, tgt_mod)
        err = (base[:, :-1] - out3[:, :-1]).abs().max().item()
        print(f"[3] causality max|diff| = {err:.2e}  "
              f"{'PASS' if err < 1e-3 else 'FAIL (decoder sees the future)'}")


def check_overfit_one_batch(model, batch, meta, device, steps=200, lr=1e-3):
    m = copy.deepcopy(model).to(device)
    m.eval()                                          # eval = no dropout; grads still work
    b = to_device({k: v[:16] for k, v in batch.items()}, device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    V = meta["vocab_size"]
    for i in range(1, steps + 1):
        loss = F.cross_entropy(m(b["src_ids"], b["tgt_ids"]).reshape(-1, V),
                               b["labels"].reshape(-1), ignore_index=IGNORE_INDEX)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if i % 50 == 0 or i == 1:
            print(f"[4] overfit step {i:3d} loss {loss.item():.4f}")
    print(f"[4] final {loss.item():.4f} -> "
          f"{'PASS' if loss.item() < 0.1 else 'WEAK' if loss.item() < 0.5 else 'FAIL'}")


# ----------------------------- checkpoints -----------------------------
def save_ckpt(path, model, optimizer, scheduler, scaler, cfg, meta, epoch, step, best, bad):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "cfg": asdict(cfg), "meta": meta, "epoch": epoch, "step": step,
        "best_score": best, "bad_epochs": bad,
    }, path)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run sanity checks and exit")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--amp", type=str)
    ap.add_argument("--monitor", type=str)
    args = ap.parse_args()

    cfg = TrainConfig()
    for k in ("epochs", "batch_size", "lr", "amp", "monitor"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(cfg, device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    loaders, meta, tok = build_dataloaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.seed
    )
    model = build_model(cfg, meta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M amp={amp_dtype} "
          f"vocab={meta['vocab_size']} mx_sq_ln={meta['mx_sq_ln']}")

    if args.check:
        batch = next(iter(loaders["train"]))
        check_masks_and_init(model, batch, meta, device)
        check_overfit_one_batch(model, batch, meta, device)
        return

    total_steps = cfg.epochs * len(loaders["train"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    start_epoch, step, best, bad = 1, 0, -float("inf"), 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        best, bad = ck["best_score"], ck["bad_epochs"]
        print(f"resumed from {args.resume} at epoch {ck['epoch']}")

    log_path = out_dir / "metrics.jsonl"
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_loss, step = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, scaler, cfg,
            device, amp_dtype, epoch, step,
        )
        val_nll, val_ppl = evaluate_loss(model, loaders["val"], device, amp_dtype)
        gen, samples = evaluate_generation(
            model, loaders["val"], tok, meta, cfg, device, amp_dtype
        )

        score = -val_nll if cfg.monitor == "val_loss" else gen["em"]   # higher = better
        improved = score > best
        if improved:
            best, bad = score, 0
            save_ckpt(out_dir / "best.pt", model, optimizer, scheduler, scaler,
                      cfg, meta, epoch, step, best, bad)
        else:
            bad += 1

        row = {"epoch": epoch, "step": step, "train_loss": train_loss,
               "val_nll": val_nll, "val_ppl": val_ppl, "val_em": gen["em"],
               "val_tool_acc": gen["tool_acc"], "lr": scheduler.get_last_lr()[0],
               "secs": time.time() - t0, "best": improved}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch:02d} | train {train_loss:.3f} | val_nll {val_nll:.3f} "
              f"(ppl {val_ppl:.1f}) | EM {gen['em']:.3f} | tool {gen['tool_acc']:.3f} "
              f"| {row['secs']:.0f}s {'*' if improved else ''}")
        if epoch % cfg.show_samples_every == 0:
            print_samples(samples)

        save_ckpt(out_dir / "last.pt", model, optimizer, scheduler, scaler,
                  cfg, meta, epoch, step, best, bad)
        if bad >= cfg.patience:
            print(f"early stopping: no improvement for {cfg.patience} epochs")
            break

    # final test with the best checkpoint
    ck = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    test_nll, test_ppl = evaluate_loss(model, loaders["test"], device, amp_dtype)
    gen, samples = evaluate_generation(
        model, loaders["test"], tok, meta, cfg, device, amp_dtype, n_samples=5
    )
    result = {"best_epoch": ck["epoch"], "test_nll": test_nll, "test_ppl": test_ppl,
              "test_em": gen["em"], "test_tool_acc": gen["tool_acc"], "n": gen["n"]}
    (out_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print("TEST:", result)
    print_samples(samples)


if __name__ == "__main__":
    main()
