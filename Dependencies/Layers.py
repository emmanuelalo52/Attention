import torch.nn as nn
from GeLU import TanhGelu
import Linear
class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.cfc = Linear(emb_dim, 4*emb_dim)
        self.gelu = TanhGelu()
        self.c_proj = Linear(4 * emb_dim, emb_dim)
    def forward(self,x):
            x = self.cfc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            return x