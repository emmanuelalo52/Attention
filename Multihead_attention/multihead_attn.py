import numpy as np
import torch
from ..Architecture.LayerNorm import LayerNorm
from ..Architecture.Linear import Linear
from ..Architecture.dropout import Dropout
from ..Architecture.Softmax import Softmax
from ..Architecture.ReLU import ReLU
import torch.nn as nn
from torch.nn import functional as F


#Simple self attention mechanism
class SelfAttention(nn.Module):
    def __init__(self,n_embed,head_size,block_size, dropout=0.1):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        q = self.q_l(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_l(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_l(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        wei= q @ k.transpose(-2,-1)*C**-0.5# switch the last 2 dimensions(T,C). We then dot multiply (B,16,T) @ (B,T,16) to give us (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out
    
#Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size, n_embed,dropout=0.1):
        super().__init__()
        self.n_heads = nn.ModuleList(SelfAttention(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.n_heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
#FFN(x) = max(0, xW1 + b1)W2 + b2
class PositionWiseForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed,n_embed),
            nn.Dropout(),
        )
    def forward(self,x):
        return self.net(x)
