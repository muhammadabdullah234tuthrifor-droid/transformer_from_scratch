from torch import nn

from decoder import Decoder
from encoder import Encoder


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        mx_sq_ln,
        inr_dim,
        lnr_dim,
        nm_hds,
        nm_lyrs,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            mx_sq_ln=mx_sq_ln,
            inr_dim=inr_dim,
            lnr_dim=lnr_dim,
            nm_hds=nm_hds,
            nm_lyrs=nm_lyrs,
            dropout=dropout,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            mx_sq_ln=mx_sq_ln,
            inr_dim=inr_dim,
            lnr_dim=lnr_dim,
            nm_hds=nm_hds,
            nm_lyrs=nm_lyrs,
            dropout=dropout,
        )

    def forward(
        self,
        src_ids,
        tgt_ids,
        pad_tok_id,
        encd_padding_mask,
        dcd_padding_mask=False,
        casual_mask=True,
        crs_atn_mask=True,
    ):
        encd_oput, _attn_weights_per_layer = self.encoder(
            src_ids=src_ids, pad_tok_id=pad_tok_id, padding_mask=encd_padding_mask
        )
        oput, _self_wts_pr_lyr, _crs_wts_pr_lyr = self.decoder(
            encd_oput=encd_oput,
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            pad_tok_id=pad_tok_id,
            padding_mask=dcd_padding_mask,
            casual_mask=casual_mask,
            crs_atn_mask=crs_atn_mask,
        )
        return oput
