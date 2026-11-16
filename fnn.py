

import torch
from torch import nn
import torch.nn.functional as F



class SwihFNN(nn.Module):
    def __init__(self, inr_dim, lnr_dim):
        super().__init__()
        self.W_DN = nn.Linear(lnr_dim, inr_dim)
        self.W_GT = nn.Linear(inr_dim, lnr_dim)
        self.W_UP = nn.Linear(inr_dim, lnr_dim)

    def forward(self, x):
        return self.W_DN(F.silu(self.W_GT(x)) * self.W_UP(x))
