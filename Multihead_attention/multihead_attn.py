import numpy as np
import torch
from ..Architecture.LayerNorm import LayerNorm
from ..Architecture.Linear import Linear
from ..Architecture.dropout import Dropout
from ..Architecture.Softmax import Softmax
from ..Architecture.ReLU import ReLU
import torch.nn as nn
from torch.nn import functional as F
class MultiheadAttn(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_size = emb_dim // num_heads
        
        self.q_l = nn.Linear(emb_dim, emb_dim)
        self.k_l = nn.Linear(emb_dim, emb_dim)
        self.v_l = nn.Linear(emb_dim, emb_dim)
        self.out_linear = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        
        # Linear projections and reshape
        q = self.q_l(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_l(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_l(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # Attention
        weight = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        
        # Apply attention to values
        out = weight @ v
        
        # Reshape and apply final linear layer
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_linear(out)
        
        # Apply dropout (if needed)
        out = self.dropout(out)
        
        return out
    

#FFN(x) = max(0, xW1 + b1)W2 + b2
class PositionWiseForward(nn.Module):
    def __init__(self,input,output_size,dropout=0.1):
        super().__init__()
        self.resid_net = nn.Sequential(Linear(input, output_size),
                                       nn.ReLU(),
                                       nn.Linear(input,output_size),
                                       Dropout(dropout),)
    def forward(self,x):
        return self.resid_net(x)
