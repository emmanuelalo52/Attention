import numpy as np
import torch
from ..Dependencies.LayerNorm import LayerNorm
from ..Dependencies.Linear import Linear
from ..Dependencies.dropout import Dropout
from ..Dependencies.Softmax import Softmax
from ..Dependencies.ReLU import ReLU
import torch.nn as nn
from torch.nn import functional as F
from ..Dependencies import Scaleddotproduct

#Simple self attention mechanism


class MultiHeadAttention(nn.Module):
    def __init__(self,emb_dim,n_head,block_size):
        super().__init__()
        assert emb_dim % n_head == 0
        self.c_attn = Linear(emb_dim,3 * emb_dim)
        self.proj = Linear(emb_dim, emb_dim)
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.block_size = block_size
        self.register_buffer("bias", torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size))
    def forward(self, x):
        B,T,C = x.size()
        qkv= self.c_attn(x)
        q,k,v = qkv.split(self.emb_dim,dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        scores = Scaleddotproduct(q,k,v)
        weights = scores.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        weights = self.proj(weights)
        return weights
#FFN(x) = max(0, xW1 + b1)W2 + b2
class PositionWiseForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            Linear(n_embed,4 * n_embed),
            ReLU(),
            Linear(4 * n_embed,n_embed),
            Dropout(),
        )
    def forward(self,x):
        return self.net(x)
