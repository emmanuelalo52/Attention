import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * x_norm + self.beta
        return self.out