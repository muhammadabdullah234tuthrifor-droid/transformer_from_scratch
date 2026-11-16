import torch
from torch import nn
import math
class RotationlPositionalEncoding(nn.Module):
    def __init__(self, mx_sq_ln, per_head_dim):
        super().__init__()
        frequencies = torch.exp(
            torch.arange(0, per_head_dim, 2) * (-math.log(10000.0) / per_head_dim)
        )
        positions = torch.arange(0, mx_sq_ln, dtype=torch.float32)


        angles= torch.outer(positions, frequencies)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    def forward(self, q, k):
        sq_ln_q= q.size(-2)
        sq_ln_k= k.size(-2)
        cos_q= self.cos[:sq_ln_q, :].to(q.dtype) # type: ignore
        sin_q= self.sin[:sq_ln_q, :].to(q.dtype) # type: ignore
        cos_q= torch.cat([cos_q, cos_q], dim= -1)
        sin_q= torch.cat([sin_q, sin_q], dim= -1)

        cos_k= self.cos[:sq_ln_k, :].to(k.dtype) # type: ignore
        sin_k= self.sin[:sq_ln_k, :].to(k.dtype) # type: ignore
        cos_k= torch.cat([cos_k, cos_k], dim= -1)
        sin_k= torch.cat([sin_k, sin_k], dim= -1)
        q_rotated= (q * cos_q) + (self.rotate_half(q) * sin_q)
        k_rotated= (k * cos_k) + (self.rotate_half(k) * sin_k)
        return q_rotated, k_rotated
