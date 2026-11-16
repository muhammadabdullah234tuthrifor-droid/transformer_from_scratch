import torch
from torch import nn
from attention import MultiHeadAttention
from normalization import RMSNorm
from fnn import SwihFNN
class Transformer(nn.Module):
    def __init__(self, mx_sq_ln, inr_dim, nm_hds, lnr_dim, use_crs_atn: bool= False, eps=1e-6):
        super().__init__()
        self.norm_1 = RMSNorm(inr_dim=inr_dim, eps=eps)
        self.norm_2 = RMSNorm(inr_dim=inr_dim, eps=eps)
        self.dropout= nn.Dropout(0.1)
        self.atn = MultiHeadAttention(mx_sq_ln= mx_sq_ln, inner_dimension=inr_dim, attention_heads=nm_hds)
        self.use_crs_atn= use_crs_atn
        if self.use_crs_atn:
            self.crs_atn = MultiHeadAttention(
            mx_sq_ln=mx_sq_ln, inner_dimension=inr_dim, attention_heads=nm_hds,
            use_rope= False
        )
            self.norm_3 = RMSNorm(inr_dim=inr_dim, eps=eps)

        else:
            self.crs_atn= None
            self.norm_3= None

        self.fnn = SwihFNN(inr_dim=inr_dim, lnr_dim=lnr_dim)

    @staticmethod
    def casual_mask(sq_ln: int, device):
        return torch.tril(torch.ones(sq_ln, sq_ln, device=device))

    @staticmethod
    def batch_mask(tok_ids, pad_tok_id):
        mask = (tok_ids != pad_tok_id).float()
        return mask.unsqueeze(1).unsqueeze(2)

    @staticmethod


    def combined_mask(tok_ids, pad_tok_id, device):
        batch_size, seq_len = tok_ids.shape

        # True means the token can be used as an attention key.
        valid_tokens = tok_ids.ne(pad_tok_id).clone()

        # The first decoder position is the start position, even when its
        # token ID equals PAD_ID.
        if seq_len > 0:
            valid_tokens[:, 0] = True

        causal = Transformer.casual_mask(
            sq_ln=seq_len,
            device=device,
        ).bool()

        # Shape: (batch, 1, 1, seq_len)
        key_mask = valid_tokens[:, None, None, :].to(device=device)

        # Shape: (batch, 1, seq_len, seq_len)
        mask = causal[None, None, :, :] & key_mask

        return mask
    # def combined_mask(tok_ids, pad_tok_id, device):
    #     sq_ln = tok_ids.size(1)
    #     casual_mask = Transformer.casual_mask(sq_ln=sq_ln, device=device)
    #     batch_mask = Transformer.batch_mask(tok_ids=tok_ids, pad_tok_id=pad_tok_id)
    #     return casual_mask.unsqueeze(0).unsqueeze(0) * batch_mask


    def forward(
        self, x,  padding_mask, casual_mask= False, crs_atn_mask= False, encd_oput= None, tgt_ids= None, src_ids= None, pad_tok_id=None, crs_atn= False
    ):
        device = x.device
        if casual_mask:
            seq_len = tgt_ids.size(1) #type: ignore
            mask = Transformer.casual_mask(
                    sq_ln=seq_len,
                    device=device,
                ).unsqueeze(0).unsqueeze(0)
            # mask = Transformer.combined_mask(tok_ids= tgt_ids, pad_tok_id= pad_tok_id, device=device) # decoder casual attention(tgt_ids will be passed as tok_ids)
        elif padding_mask:
            mask = Transformer.batch_mask(tok_ids= src_ids, pad_tok_id= pad_tok_id) # encoder case(src_ids will be passed as tok_ids)
        else:
            mask= None

        crs_mask= None

        if self.use_crs_atn and crs_atn_mask:
            crs_mask= Transformer.batch_mask(tok_ids= src_ids, pad_tok_id= pad_tok_id) # decoder corss attention(src_ids will be passed as tok_ids)



        norm_x= self.norm_1(x)
        oput, slf_atn_wts= self.atn(norm_x, norm_x, norm_x, mask= mask)
        h= x + self.dropout(oput)
        oput= self.norm_2(h)
        crs_atn_wts= []
        if self.use_crs_atn and crs_atn:
            oput, crs_atn_wts= self.crs_atn(oput, encd_oput, encd_oput, mask= crs_mask)
            h= h + self.dropout(oput)
            oput= self.norm_3(h)
        fnn_out= self.fnn(oput)
        oput = h + self.dropout(fnn_out)
        return oput, slf_atn_wts, crs_atn_wts
