import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import ByteLevelBPETokenizer

# ----------------------------- config -----------------------------
DATASET_ID = "mecha-org/linux-command-dataset"
OUT_DIR = Path("processed_linux_cmd")
SEED = 42
VOCAB_SIZE = 4000          # small data -> small vocab
MIN_FREQ = 2
MAX_LEN_CAP = 128          # examples longer than this (in tokens) are dropped
VAL_FRAC = 0.05
TEST_FRAC = 0.05
SPLIT_BY = "command"       # "command" (honest) or "row" (easy, leaky)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]   # ids 0, 1, 2, 3

# ----------------------------- 1. load -----------------------------
def pick_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {columns}")


def load_raw():
    ds = load_dataset(DATASET_ID)
    pairs = []
    for split_name, split in ds.items():
        cols = split.column_names
        src_col = pick_column(cols, ["input", "instruction", "prompt", "question"])
        tgt_col = pick_column(cols, ["output", "command", "response", "answer"])
        print(f"split={split_name!r} rows={len(split)} columns={cols} "
              f"-> using src={src_col!r}, tgt={tgt_col!r}")
        for row in split:
            pairs.append((row[src_col], row[tgt_col]))
    return pairs

# ----------------------------- 2. clean -----------------------------
_CTRL_ALL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
_PROMPT = re.compile(r"^\$\s+")


def clean_instruction(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = _CTRL_ALL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def clean_command(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL_ALL.sub("", text).strip()
    text = _PROMPT.sub("", text)                      # "$ ls" -> "ls"
    return "\n".join(line.rstrip() for line in text.split("\n"))


def clean_and_dedupe(raw_pairs):
    seen, pairs, bad = set(), [], 0
    for s, t in raw_pairs:
        if not isinstance(s, str) or not isinstance(t, str):
            bad += 1
            continue
        s, t = clean_instruction(s), clean_command(t)
        if not s or not t or (s, t) in seen:
            bad += 1
            continue
        seen.add((s, t))
        pairs.append((s, t))
    print(f"raw={len(raw_pairs)} kept={len(pairs)} removed(empty/dup/non-str)={bad}")

    by_instr = {}
    for s, t in pairs:
        by_instr.setdefault(s, set()).add(t)
    ambiguous = sum(len(v) > 1 for v in by_instr.values())
    print(f"instructions with >1 different command (label noise): {ambiguous}")
    return pairs

# ----------------------------- 3. split -----------------------------
def split_pairs(pairs):
    rng = random.Random(SEED)
    out = {"train": [], "val": [], "test": []}

    if SPLIT_BY == "command":
        groups = sorted({t for _, t in pairs})
        rng.shuffle(groups)
        n_test = int(len(groups) * TEST_FRAC)
        n_val = int(len(groups) * VAL_FRAC)
        test_g = set(groups[:n_test])
        val_g = set(groups[n_test:n_test + n_val])
        for s, t in pairs:
            name = "test" if t in test_g else "val" if t in val_g else "train"
            out[name].append((s, t))
    else:
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        n_test = int(len(shuffled) * TEST_FRAC)
        n_val = int(len(shuffled) * VAL_FRAC)
        out["test"] = shuffled[:n_test]
        out["val"] = shuffled[n_test:n_test + n_val]
        out["train"] = shuffled[n_test + n_val:]

    print({k: len(v) for k, v in out.items()})
    return out

# ----------------------------- 4. tokenizer -----------------------------
def train_tokenizer(train_pairs):
    tok = ByteLevelBPETokenizer(add_prefix_space=True)

    def corpus():
        for s, t in train_pairs:        # TRAIN split only: no leakage
            yield s
            yield t

    tok.train_from_iterator(
        corpus(), vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
    )
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(name) == i, f"{name} should have id {i}"
    print("tokenizer vocab size:", tok.get_vocab_size())
    return tok

# ----------------------------- 5. encode -----------------------------
def report_lengths(name, src_lens, tgt_lens):
    for label, lens in (("src", src_lens), ("tgt", tgt_lens)):
        p = np.percentile(lens, [50, 95, 99, 100])
        print(f"[{name}] {label} tokens p50={p[0]:.0f} p95={p[1]:.0f} "
              f"p99={p[2]:.0f} max={p[3]:.0f}")


def encode_split(tok, pairs, name):
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    src_enc = tok.encode_batch([s for s, _ in pairs])
    tgt_enc = tok.encode_batch([t for _, t in pairs])

    rows, src_lens, tgt_lens, dropped = [], [], [], 0
    for (s_text, t_text), s, t in zip(pairs, src_enc, tgt_enc):
        src_ids = s.ids + [eos]
        tgt_in = [bos] + t.ids          # decoder input
        labels = t.ids + [eos]          # what the decoder must predict
        src_lens.append(len(src_ids))
        tgt_lens.append(len(tgt_in))
        if len(src_ids) > MAX_LEN_CAP or len(tgt_in) > MAX_LEN_CAP:
            dropped += 1
            continue
        rows.append({
            "src_ids": src_ids, "tgt_ids": tgt_in, "labels": labels,
            "src_text": s_text, "tgt_text": t_text,
        })
    report_lengths(name, src_lens, tgt_lens)
    print(f"[{name}] kept={len(rows)} dropped_too_long={dropped}")
    return rows

# ----------------------------- 6. sanity + save -----------------------------
def roundtrip_check(tok, rows, n=500):
    bad = 0
    for r in rows[:n]:
        decoded = tok.decode(r["labels"][:-1], skip_special_tokens=True).strip()
        bad += decoded != r["tgt_text"]
    print(f"round-trip mismatches in first {min(n, len(rows))} train rows: {bad}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pairs = clean_and_dedupe(load_raw())
    splits = split_pairs(pairs)
    tok = train_tokenizer(splits["train"])

    encoded = {name: encode_split(tok, p, name) for name, p in splits.items()}
    roundtrip_check(tok, encoded["train"])

    longest = max(
        max(len(r["src_ids"]), len(r["tgt_ids"]))
        for rows in encoded.values() for r in rows
    )
    mx_sq_ln = ((longest + 7) // 8) * 8        # round up to a multiple of 8

    DatasetDict({k: Dataset.from_list(v) for k, v in encoded.items()}) \
        .save_to_disk(str(OUT_DIR / "dataset"))
    tok.save(str(OUT_DIR / "tokenizer.json"))

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "pad_tok_id": tok.token_to_id("<pad>"),
        "bos_tok_id": tok.token_to_id("<bos>"),
        "eos_tok_id": tok.token_to_id("<eos>"),
        "mx_sq_ln": mx_sq_ln,
        "split_by": SPLIT_BY,
        "sizes": {k: len(v) for k, v in encoded.items()},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved ->", OUT_DIR, meta)


if __name__ == "__main__":
    main()
