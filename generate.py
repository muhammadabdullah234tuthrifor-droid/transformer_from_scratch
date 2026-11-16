import torch
from torch import nn
from enc_dcd import EncoderDecoderTransformer
class Seq2SeqLM(nn.Module):
    """EncoderDecoderTransformer + output head + greedy generation."""

    def __init__(self, vocab_size, mx_sq_ln, inr_dim, lnr_dim, nm_hds, nm_lyrs,
                 pad_tok_id, dropout=0.1, tie_embeddings=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.mx_sq_ln = mx_sq_ln
        self.pad_tok_id = pad_tok_id

        self.core = EncoderDecoderTransformer(
            src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
            mx_sq_ln=mx_sq_ln, inr_dim=inr_dim, lnr_dim=lnr_dim,
            nm_hds=nm_hds, nm_lyrs=nm_lyrs, dropout=dropout,
        )
        self.head = nn.Linear(inr_dim, vocab_size, bias=False)

        if tie_embeddings:
            # one matrix shared by decoder input, output head, and encoder input
            self.head.weight = self.core.decoder.embedding.weight
            self.core.encoder.embedding.weight = self.core.decoder.embedding.weight

    def forward(self, src_ids, tgt_ids):
        hidden = self.core(
            src_ids=src_ids, tgt_ids=tgt_ids, pad_tok_id=self.pad_tok_id,
            encd_padding_mask=True, casual_mask=True, crs_atn_mask=True,
        )
        return self.head(hidden)                       # (B, T, vocab)

    @torch.no_grad()
    def generate(self, src_ids, bos_id, eos_id, max_new_tokens):
        was_training = self.training
        self.eval()
        B, device = src_ids.size(0), src_ids.device
        max_new_tokens = min(max_new_tokens, self.mx_sq_ln - 1)

        enc_out, _ = self.core.encoder(
            src_ids=src_ids, pad_tok_id=self.pad_tok_id, padding_mask=True
        )                                              # encode ONCE
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            hidden, _, _ = self.core.decoder(
                encd_oput=enc_out, src_ids=src_ids, tgt_ids=ys,
                pad_tok_id=self.pad_tok_id, casual_mask=True,
                padding_mask=False, crs_atn_mask=True,
            )
            next_tok = self.head(hidden[:, -1]).argmax(dim=-1)
            next_tok = next_tok.masked_fill(done, self.pad_tok_id)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            done |= next_tok.eq(eos_id)
            if bool(done.all()):
                break

        self.train(was_training)
        return ys[:, 1:]                               # drop <bos>
