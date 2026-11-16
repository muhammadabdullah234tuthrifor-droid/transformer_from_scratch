import torch
from torch import nn
class RMSNorm(nn.Module):
    def __init__(self, inr_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(inr_dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )  # same old-school MSE
        return self.gamma * (x / rms)
