import torch
from torch import nn
from transformer import Transformer
from normalization import RMSNorm
import math
class Decoder(nn.Module):
    def __init__(
        self, vocab_size, mx_sq_ln, inr_dim, lnr_dim, nm_hds, nm_lyrs, dropout=0.1
    ):
        super().__init__()
        self.inr_dim = inr_dim
        self.lnr_dim = lnr_dim
        self.mx_sq_ln = mx_sq_ln
        self.nm_hds = nm_hds
        self.norm = RMSNorm(inr_dim=inr_dim)
        self.embedding = nn.Embedding(vocab_size, inr_dim)
        nn.init.normal_(self.embedding.weight, mean= 0.0, std= self.inr_dim **-0.5)
        self.pos_embedding = nn.Embedding(mx_sq_ln, inr_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)


        self.dropout = nn.Dropout(dropout)

        self.lyrs = nn.ModuleList(
            [
                Transformer(
                    mx_sq_ln=self.mx_sq_ln,
                    inr_dim=self.inr_dim,
                    lnr_dim=self.lnr_dim,
                    nm_hds=self.nm_hds,
                    use_crs_atn= True
                )
                for _ in range(nm_lyrs)
            ]
        )

    def forward(self, encd_oput, src_ids, tgt_ids, pad_tok_id, casual_mask, padding_mask= False, crs_atn_mask= True, crs_atn= True):
        positions = torch.arange(tgt_ids.size(1), device=tgt_ids.device)
        x = self.embedding(tgt_ids) * math.sqrt(self.inr_dim) + self.pos_embedding(positions)
        x = self.dropout(x)
        slf_atn_wts_pr_lyr = []
        crs_atn_wts_pr_lyr = []
        for lyr in self.lyrs:
            x, slf_atn_wts, crs_atn_wts = lyr(
                x=x,
                encd_oput=encd_oput,
                src_ids=src_ids,
                tgt_ids= tgt_ids,
                pad_tok_id=pad_tok_id,
                crs_atn=True,
                padding_mask= padding_mask,
                casual_mask= casual_mask,
                crs_atn_mask= crs_atn_mask
            )
            slf_atn_wts_pr_lyr.append(slf_atn_wts)
            crs_atn_wts_pr_lyr.append(crs_atn_wts)
        oput = self.norm(x)

        return oput, slf_atn_wts_pr_lyr, crs_atn_wts_pr_lyr
