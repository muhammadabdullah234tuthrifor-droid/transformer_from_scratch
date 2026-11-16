import torch
from torch import nn
from transformer import Transformer
from normalization import RMSNorm
import math
class Encoder(nn.Module):
    def __init__(
        self, vocab_size, mx_sq_ln, inr_dim, lnr_dim, nm_hds, nm_lyrs, dropout=0.1
    ):
        super().__init__()
        self.inr_dim = inr_dim
        self.nm_hds = nm_hds
        self.lnr_dim = lnr_dim
        self.per_head_dim = int(self.inr_dim / self.nm_hds)
        self.embedding = nn.Embedding(vocab_size, inr_dim)
        nn.init.normal_(self.embedding.weight, mean= 0.0, std= self.inr_dim **-0.5)

        self.pos_embedding = nn.Embedding(mx_sq_ln, inr_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout)

        self.lyrs = nn.ModuleList(
            [
                Transformer(
                    mx_sq_ln=mx_sq_ln,
                    inr_dim=self.inr_dim,
                    nm_hds=self.nm_hds,
                    lnr_dim=self.lnr_dim,
                    use_crs_atn= False
                )
                for _ in range(nm_lyrs)
            ]
        )
        self.norm = RMSNorm(inr_dim=inr_dim)

    def forward(self, src_ids, pad_tok_id, padding_mask):
        positions = torch.arange(src_ids.size(1), device=src_ids.device)
        x = self.embedding(src_ids) * math.sqrt(self.inr_dim) + self.pos_embedding(positions)
        # x = self.embedding(src_ids) * math.sqrt(self.inr_dim)
        x = self.dropout(x)
        atn_wts_pr_lyr = []
        for lyr in self.lyrs:
            x, atn_wts, _ = lyr(
                x=x, src_ids=src_ids, pad_tok_id=pad_tok_id, padding_mask=padding_mask
            )
            atn_wts_pr_lyr.append(atn_wts)
        oput = self.norm(x)
        return oput, atn_wts_pr_lyr
