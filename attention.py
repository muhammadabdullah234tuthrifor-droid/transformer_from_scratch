import torch
from torch import nn
from encoding import RotationlPositionalEncoding
import math
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, mx_sq_ln, inner_dimension: int = 512, attention_heads: int = 8, use_rope: bool= True):
        super().__init__()
        assert inner_dimension % attention_heads == 0, (
            "hidden dimension should be equally divided among attention heads"
        )
        self.inner_dimension = inner_dimension
        self.attention_heads = attention_heads
        self.per_head_dim = self.inner_dimension // attention_heads
        self.W_Q = nn.Linear(inner_dimension, inner_dimension)
        self.W_K = nn.Linear(inner_dimension, inner_dimension)
        self.W_V = nn.Linear(inner_dimension, inner_dimension)
        self.W_O = nn.Linear(inner_dimension, inner_dimension)
        self.encoding = RotationlPositionalEncoding(
            mx_sq_ln=mx_sq_ln, per_head_dim=self.per_head_dim
        ) if use_rope else None

    def split_heads(self, x, batch_size: int = 64):
        x = x.view(batch_size, -1, self.attention_heads, self.per_head_dim)
        return x.transpose(-2, -3)

    def combine_heads(self, x, batch_size):
        x = x.transpose(-3, -2).contiguous()
        x = x.view(batch_size, -1, self.inner_dimension)
        return x

    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)
        Q = self.split_heads(self.W_Q(q), batch_size=batch_size)
        K = self.split_heads(self.W_K(k), batch_size=batch_size)
        V = self.split_heads(self.W_V(v), batch_size=batch_size)
        if self.encoding is not None:
            Q, K = self.encoding(Q, K)
        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.per_head_dim)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        scores = F.softmax(logits, dim=-1)
        weights = torch.matmul(scores, V)
        weights = self.combine_heads(weights, batch_size=batch_size)
        output = self.W_O(weights)
        return output, scores
