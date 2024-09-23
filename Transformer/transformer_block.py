from ..Multihead_attention.multihead_attn import *
from ..Architecture.LayerNorm import LayerNorm
import torch.nn as nn
#Transformer block
#Here computation and communication between tokens happen
# Add normalization before the feed-forward
class TransformerBlock(nn.Module):
    def __init__(self,n_embed,n_heads):
        super().__init__()
        head_size = n_embed//n_heads
        self.sa = MultiHeadAttention(n_heads,head_size)
        self.ffd = PositionWiseForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x