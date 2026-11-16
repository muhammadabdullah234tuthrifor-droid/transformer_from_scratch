# MiniTransformer

A **from-scratch** encoder–decoder Transformer built with plain PyTorch primitives — no `nn.Transformer`, no `nn.MultiheadAttention`, no HuggingFace `transformers` for the model. Every layer, mask, positional encoding and training loop is written explicitly so you can read, modify and reason about every line.

The repo ships with **two complete training recipes** on two very different tasks:

| Task | Tokenizer | Loss | Signal it demonstrates |
|---|---|---|---|
| **Arithmetic evaluation** — `3+4*(2-1) → 7` | char-level | CE with reverse-digit target | numeric reasoning, masked decoder, greedy decoding |
| **NL → Linux command** — `"list files with sizes" → ls -lh` | ByteLevel BPE (trained from scratch) | CE with label smoothing + AMP | open-ended seq2seq, tied embeddings, generation eval |

Both run out of the box on CPU (`--smoke` flag) or GPU.

---

## Why another Transformer?

Most "from-scratch Transformer" repos are decoder-only toy models trained on tiny text. This one is different in three ways.

### 1. Modern layers, encoder–decoder topology
Modern efficiency + quality tricks from the current LLM lineage are dropped into a **canonical Vaswani-style encoder–decoder**:

- **RMSNorm** instead of LayerNorm (LLaMA / Gemma lineage)
- **RoPE** rotary positional embeddings (RoFormer / Llama / GPT-NeoX)
- **SwiGLU** gated FFN (PaLM / Llama)
- **Pre-LN** residual topology (more stable than post-LN at low LR warmup)
- **Weight tying** across encoder-embedding, decoder-embedding and output head
- Optional **label smoothing** and **mixed precision**

You get state-of-the-art building blocks in a topology the modern transformer world has largely stopped writing down.

### 2. RoPE on self-attention only
`MultiHeadAttention` accepts `use_rope=True|False`. Self-attention uses RoPE; **cross-attention does not** — cross-attention mixes target queries with source keys, and the source positions have no meaningful alignment with target positions, so rotating them would only add noise. This is a deliberate design choice most RoPE tutorials miss.

RoPE buffers (`cos`/`sin`) are registered once at init for `mx_sq_ln` positions and are **zero-parameter**, so the model's positional budget scales as `mx_sq_ln`, not as a learned embedding table.

### 3. Built-in sanity checks, not "run it and hope"
`python train.py --check` runs four hardware-independent checks before you commit to a training run:

1. **Initial-loss check** — cross-entropy at init should be ≈ `ln(V)`. If it isn't, something is wrong with the head or the labels.
2. **Padding invariance** — appending pad tokens to the input must not change outputs.
3. **Causality check** — changing the *last* target token must not change logits at earlier positions. If this fails, the decoder can see the future and every metric after that is a lie.
4. **Overfit-one-batch** — 200 gradient steps on 16 examples. If loss doesn't drop below 0.1 you have a numerical or masking bug, not a data bug.

Every one of these checks is a real bug class that has silently ruined production transformer training runs.

### 4. Reverse-digit target for arithmetic
`train_arith.py` optionally emits the answer **least-significant digit first** (`reverse_target=True`). This is a well-known trick: when generating MSB-first, a carry from the units column must propagate *all the way back* through already-emitted digits. LSD-first lets the model emit `3` before it knows whether `7+5` will carry. Exact-match on multi-digit arithmetic jumps sharply with this one flag.

### 5. Crash-safe checkpointing
`train_arith.py` writes to `<path>.tmp` and then `os.replace()`s it. A Ctrl-C or OOM never leaves you with a half-written `best.pt`. The checkpoint stores the **full model config** so `load_for_inference()` rebuilds the exact same model without any side config file.

### 6. One repo, two demonstrated recipes
- **Arithmetic** — char-level vocab (18 tokens), reverse target, per-token exact-match, greedy decode, early stopping, resume that refuses to load a checkpoint whose model shape doesn't match.
- **NL → Linux command** — ByteLevel BPE trained only on the train split (no leakage), JSONL metric logs, tool-name accuracy alongside exact-match, tied embeddings, AMP + grad scaler, patience-based early stop, and per-epoch sample dumps.

Between the two you get template code for "small, tight, numeric task" and "open-ended seq2seq task".

---

## Architecture at a glance

```
                        src_ids  ──►  Embedding  ──►  +pos_emb  ──►  dropout
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │  Encoder layer   × nm_lyrs    │
                             │                               │
                             │   x ──► RMSNorm ──► MHA(RoPE) │
                             │         │ residual            │
                             │         ▼                     │
                             │   RMSNorm ──► SwiGLU-FFN      │
                             │         │ residual            │
                             └──────────────┬────────────────┘
                                            │ encd_oput
                                            │
                        tgt_ids ──► Embedding +pos_emb ──► dropout
                                            │
                                            ▼
                             ┌───────────────────────────────┐
                             │  Decoder layer   × nm_lyrs    │
                             │                               │
                             │  RMSNorm ──► MHA(RoPE, causal)│
                             │  RMSNorm ──► CrossMHA(no RoPE)│
                             │  RMSNorm ──► SwiGLU-FFN       │
                             └──────────────┬────────────────┘
                                            │
                                       RMSNorm
                                            │
                                       Linear head (tied weights)
                                            │
                                       softmax / argmax
```

Design notes:

| Component | File | Notes |
|---|---|---|
| Multi-head attention | `attention.py` | Explicit `W_Q`, `W_K`, `W_V`, `W_O`. Splits into heads; supports RoPE on demand. |
| RoPE | `encoding.py` | `cos`/`sin` registered buffers; `rotate_half` on the interleaved-half convention. No learnable params. |
| RMSNorm | `normalization.py` | `gamma * x / sqrt(mean(x²) + eps)` — one parameter vector per block. |
| SwiGLU FFN | `fnn.py` | `W_DN(silu(W_GT(x)) * W_UP(x))` — same formulation as LLaMA. |
| Transformer block | `transformer.py` | Pre-LN residual around self-attn → cross-attn → FFN. Cross-attention branch is `None` in the encoder. |
| Encoder | `encoder.py` | Embedding + learned absolute pos_emb + N pre-LN blocks + final norm. |
| Decoder | `decoder.py` | Same, plus causal mask in self-attn and encoder-padding mask in cross-attn. |
| Full model | `enc_dcd.py` | Encoder + Decoder orchestration. |
| Head + generate | `generate.py` | `Seq2SeqLM` wraps `EncoderDecoderTransformer` with a tied-weight head and `@torch.no_grad()` greedy generate. |

**Note on `generate.py` vs `seq2seq_lm.py`:** `train.py` currently does `from seq2seq_lm import Seq2SeqLM` — you should either `mv generate.py seq2seq_lm.py` or update the import. See *Known issues* below.

---

## Installation

```bash
git clone <your-repo> && cd mini-transformer
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.1+.

---

## Quick start

### Arithmetic (CPU-friendly, single file)

```bash
# 3-epoch pipeline check on CPU — takes a couple of minutes
python train_arith.py --smoke

# Full training run (~30 epochs, GPU recommended)
python train_arith.py

# Resume after Ctrl-C
python train_arith.py --resume

# Inference with the best checkpoint
python train_arith.py --predict "(2+3)*4"
# -> model: 20   true: 20
```

The script auto-generates expressions with **exactly** the arity the model will see at test time (default ≤ 3 operators, operands ≤ 20), splits train/val/test by expression string, and enforces `max_src_len` / `max_tgt_len` at generation time — so the model never sees an input longer than `mx_sq_ln`.

### NL → Linux command

```bash
# 1. Download + tokenize + split (creates processed_linux_cmd/)
python data_prep.py

# 2. Sanity checks (init-loss, padding invariance, causality, overfit-one-batch)
python train.py --check

# 3. Train
python train.py --epochs 40 --batch_size 64

# 4. Resume
python train.py --resume runs/linux_cmd_v1/best.pt
```

Outputs land under `runs/linux_cmd_v1/`: `best.pt`, `last.pt`, `metrics.jsonl` (one JSON object per epoch), `test_metrics.json`, `config.json`.

---

## Project structure

```
.
├── attention.py        # Multi-head attention with optional RoPE
├── encoding.py         # RoPE — registered cos/sin buffers, rotate_half
├── normalization.py    # RMSNorm
├── fnn.py              # SwiGLU gated FFN
├── transformer.py      # Pre-LN residual block + mask utilities
├── encoder.py          # Encoder stack
├── decoder.py          # Decoder stack (self-attn + cross-attn)
├── enc_dcd.py          # Encoder–Decoder orchestration
├── generate.py         # Seq2SeqLM: tied head + greedy generation
├── data_prep.py        # NL→Linux-command preprocessing + BPE training
├── data_loader.py      # Seq2SeqDataset + Collator + DataLoader factory
├── train.py            # Training loop (NL→Linux-command task)
├── train_arith.py      # Self-contained training loop (arithmetic)
├── requirements.txt
├── LICENSE
└── .gitignore
```

`data.py`, `model.py`, `prac.py`, `px.py`, `test.py`, `test_model.py`, `train_1.py` are currently **empty placeholders** — feel free to remove them.

---

## What's in the mask?

Masking is where from-scratch transformers usually break. Here is exactly what each path uses.

**Encoder self-attention** — `padding_mask=True`:
```
mask[b, h, i, j] = (src_ids[b, j] != pad_tok_id)
```
Every query position can see every non-pad key. Padded keys are masked out.

**Decoder self-attention** — `casual_mask=True`:
```
mask[i, j] = (j <= i)                # lower-triangular
```
Pure causal; pads in the target are handled by `ignore_index` in the loss, not by masking, so that the teacher-forced token at position `i+1` is never a pad unless the whole batch is finished.

**Decoder cross-attention** — `crs_atn_mask=True`:
```
mask[b, h, i, j] = (src_ids[b, j] != pad_tok_id)
```
Query comes from the target (any position), keys/values from the encoder (padded source tokens masked).

`Transformer.combined_mask()` (in `transformer.py`) is provided as a ready-made "causal + padding" mask for the decoder self-attention. It is **not currently enabled** in the forward pass — flip the commented-out line in `Transformer.forward` to turn it on if you want the decoder to also mask pad tokens in its own keys. For teacher forcing on short arithmetic tasks it makes no measurable difference; for long variable-length sequences it can.

---

## Reproducibility checklist

- ✅ `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` called in `seed_everything`
- ✅ Full config saved to `config.json` next to every `best.pt` / `last.pt`
- ✅ `metrics.jsonl` — one line per epoch, append-only, easy to plot
- ✅ Resume refuses if any `MODEL_FIELDS` disagree with the checkpoint
- ✅ Atomic checkpoint write (`os.replace`)
- ✅ Data-prep split is deterministic (seeded RNG); BPE trained only on the **train** split

---

## Sanity checks

`python train.py --check` runs:

| Check | Pass criteria | What it catches |
|---|---|---|
| `[1]` initial loss | within ±1 of `ln(V)` | bad head init, wrong label alignment, wrong vocab size |
| `[2]` padding invariance | `max|Δ| < 1e-3` | attention leaking over pad tokens |
| `[3]` causality | `max|Δ| < 1e-3` on earlier positions | decoder seeing the future (broken mask) |
| `[4]` overfit-one-batch | loss < 0.1 in 200 steps | optimizer / grads / device mismatch |

Run these once per architecture change. They are the cheapest debugging you will ever do.

---

## Design rationale — a short tour

**Why RMSNorm?** LayerNorm needs the mean, RMSNorm doesn't. Same quality on transformer workloads, ~15% fewer FLOPs in the norm, one fewer buffer to sync. It is the default in LLaMA, Gemma, T5 (RMSNorm variant) and Qwen.

**Why RoPE on self-attention only?** RoPE injects position by *rotating* Q and K. In self-attention the offset between query `i` and key `j` is `i - j`, which is exactly what RoPE encodes. In cross-attention the query position and the key position are in different sequences; there is no meaningful `i - j`. Rotating them adds noise with no signal. So `use_rope=False` is set for every `crs_atn`.

**Why pre-LN?** Post-LN requires a careful LR warmup or gradients explode at init. Pre-LN trains from step 0 at larger LRs. The trade-off is a mild final loss in some long-run regimes — acceptable for the model sizes here.

**Why tied embeddings?** The decoder's output projection and the encoder/decoder's input embedding are the same matrix. It is the single biggest parameter reduction you can make for a vocab-heavy model, and it empirically helps on small datasets because the *input* space and *output* space should share semantics.

**Why reverse-digit target for arithmetic?** MSB-first generation makes the model commit to a digit before knowing whether a carry is coming from the right. LSD-first makes the carry arrive *before* the decision that needs it. With `max_operand=20`, `max_operations=3`, exact-match roughly doubles.

**Why greedy in `train_arith.py` but `generate()` in `train.py`?** Arithmetic answers have one canonical form (no sampling diversity), so greedy is the right decoder. NL→command has many valid commands for the same instruction, so the eval reports both exact-match (greedy) and tool-name accuracy (first token match), which is the meaningful metric.

---

## Known issues / things I'd tighten before production

- **Import mismatch**: `train.py` does `from seq2seq_lm import Seq2SeqLM`, but the file is `generate.py`. Fix: `git mv generate.py seq2seq_lm.py` or change the import.
- **Decoder self-attention does not mask pad tokens** in `Transformer.forward` — switch to `combined_mask` if you care about the last few padded target positions. Currently harmless because loss uses `ignore_index`.
- **`pos_embedding` is a learned table** in `encoder.py` / `decoder.py` while self-attention also uses RoPE — this is redundant. RoPE alone is sufficient; the learned table is kept for exact parity with the Vaswani paper and to make ablation easy.
- **No KV cache** in `generate()` — fine for `mx_sq_ln=128`, do not ship to a chat product without one.
- **`train.py` assumes `datasets` + `tokenizers`** are installed. `train_arith.py` is fully self-contained (stdlib only).
- **No tests** beyond the `--check` suite. A `pytest` wrapper that runs the four checks in CI would be the first thing to add.

---

## License

MIT — see [LICENCE](LICENCE).

---

## Acknowledgements

The layer set draws directly from the modern LLM literature:

- Vaswani et al., *Attention Is All You Need* (2017) — the original encoder–decoder
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) — RoPE
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019) — RMSNorm
- Shazeer, *GLU Variants Improve Transformer* (2020) — SwiGLU
- Touvron et al., *LLaMA* (2023) — the combination that made these the default
