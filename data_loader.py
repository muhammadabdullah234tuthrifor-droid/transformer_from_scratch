import json
from pathlib import Path

import torch
from datasets import load_from_disk
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = -100


class Seq2SeqDataset(Dataset):
    def __init__(self, hf_split):
        # Load columns into plain Python lists once (tiny dataset, so cheap)
        self.src = hf_split["src_ids"]
        self.tgt = hf_split["tgt_ids"]
        self.lbl = hf_split["labels"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i], self.lbl[i]


class Collator:
    def __init__(self, pad_tok_id: int):
        self.pad_tok_id = pad_tok_id

    def __call__(self, batch):
        src, tgt, lbl = zip(*batch)
        to_t = lambda seqs: [torch.tensor(s, dtype=torch.long) for s in seqs]
        return {
            "src_ids": pad_sequence(to_t(src), batch_first=True,
                                    padding_value=self.pad_tok_id),
            "tgt_ids": pad_sequence(to_t(tgt), batch_first=True,
                                    padding_value=self.pad_tok_id),
            "labels": pad_sequence(to_t(lbl), batch_first=True,
                                   padding_value=IGNORE_INDEX),
        }


def build_dataloaders(processed_dir="processed_linux_cmd", batch_size=64,
                      num_workers=0, seed=42):
    root = Path(processed_dir)
    meta = json.loads((root / "meta.json").read_text())
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    dsd = load_from_disk(str(root / "dataset"))

    collate = Collator(meta["pad_tok_id"])
    g = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            Seq2SeqDataset(dsd[name]),
            batch_size=batch_size,
            shuffle=(name == "train"),
            collate_fn=collate,
            num_workers=num_workers,
            generator=g if name == "train" else None,
        )
        for name in ("train", "val", "test")
    }
    return loaders, meta, tok


if __name__ == "__main__":
    loaders, meta, tok = build_dataloaders(batch_size=8)
    b = next(iter(loaders["train"]))
    print({k: tuple(v.shape) for k, v in b.items()})

    # Check teacher-forcing alignment on row 0
    n = int((b["labels"][0] != IGNORE_INDEX).sum())
    assert torch.equal(b["tgt_ids"][0, 1:n], b["labels"][0, :n - 1])
    assert b["tgt_ids"][0, 0].item() == meta["bos_tok_id"]
    assert b["labels"][0, n - 1].item() == meta["eos_tok_id"]

    print("SRC :", tok.decode(b["src_ids"][0].tolist(), skip_special_tokens=True).strip())
    print("TGT :", tok.decode(b["labels"][0, :n].tolist(), skip_special_tokens=True).strip())
    print("alignment OK, pad id =", meta["pad_tok_id"], "mx_sq_ln =", meta["mx_sq_ln"])
    import json
    from pathlib import Path

    import torch
    from datasets import load_from_disk
    from tokenizers import Tokenizer
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

    IGNORE_INDEX = -100


    class Seq2SeqDataset(Dataset):
        def __init__(self, hf_split):
            # Load columns into plain Python lists once (tiny dataset, so cheap)
            self.src = hf_split["src_ids"]
            self.tgt = hf_split["tgt_ids"]
            self.lbl = hf_split["labels"]

        def __len__(self):
            return len(self.src)

        def __getitem__(self, i):
            return self.src[i], self.tgt[i], self.lbl[i]


    class Collator:
        def __init__(self, pad_tok_id: int):
            self.pad_tok_id = pad_tok_id

        def __call__(self, batch):
            src, tgt, lbl = zip(*batch)
            to_t = lambda seqs: [torch.tensor(s, dtype=torch.long) for s in seqs]
            return {
                "src_ids": pad_sequence(to_t(src), batch_first=True,
                                        padding_value=self.pad_tok_id),
                "tgt_ids": pad_sequence(to_t(tgt), batch_first=True,
                                        padding_value=self.pad_tok_id),
                "labels": pad_sequence(to_t(lbl), batch_first=True,
                                       padding_value=IGNORE_INDEX),
            }


    def build_dataloaders(processed_dir="processed_linux_cmd", batch_size=64,
                          num_workers=0, seed=42):
        root = Path(processed_dir)
        meta = json.loads((root / "meta.json").read_text())
        tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        dsd = load_from_disk(str(root / "dataset"))

        collate = Collator(meta["pad_tok_id"])
        g = torch.Generator().manual_seed(seed)
        loaders = {
            name: DataLoader(
                Seq2SeqDataset(dsd[name]),
                batch_size=batch_size,
                shuffle=(name == "train"),
                collate_fn=collate,
                num_workers=num_workers,
                generator=g if name == "train" else None,
            )
            for name in ("train", "val", "test")
        }
        return loaders, meta, tok


    if __name__ == "__main__":
        loaders, meta, tok = build_dataloaders(batch_size=8)
        b = next(iter(loaders["train"]))
        print({k: tuple(v.shape) for k, v in b.items()})

        # Check teacher-forcing alignment on row 0
        n = int((b["labels"][0] != IGNORE_INDEX).sum())
        assert torch.equal(b["tgt_ids"][0, 1:n], b["labels"][0, :n - 1])
        assert b["tgt_ids"][0, 0].item() == meta["bos_tok_id"]
        assert b["labels"][0, n - 1].item() == meta["eos_tok_id"]

        print("SRC :", tok.decode(b["src_ids"][0].tolist(), skip_special_tokens=True).strip())
        print("TGT :", tok.decode(b["labels"][0, :n].tolist(), skip_special_tokens=True).strip())
        print("alignment OK, pad id =", meta["pad_tok_id"], "mx_sq_ln =", meta["mx_sq_ln"])
