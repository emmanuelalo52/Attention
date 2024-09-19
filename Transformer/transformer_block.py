from ..Multihead_attention.multihead_attn import *
from ..Architecture.LayerNorm import LayerNorm
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ma = MultiheadAttn(n_embed, num_heads, block_size, dropout)
        self.ffd = PositionWiseForward(n_embed, 4 * n_embed, dropout)  # Typically, the hidden layer is 4x larger
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.ma(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x