import numpy as np
import torch
from ..Architecture.LayerNorm import LayerNorm
from ..Architecture.Linear import Linear
from ..Architecture.dropout import Dropout
from ..Architecture.Softmax import Softmax
from ..Architecture.ReLU import ReLU
import torch.nn as nn
class MultiheadAttn(nn.Module):
    def __init__(self,emb_dim,num_heads,dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_size = emb_dim // num_heads
        if self.head_size * num_heads == emb_dim:
            self.q_l = Linear(emb_dim,emb_dim)
            self.k_l = Linear(emb_dim,emb_dim)
            self.v_l = Linear(emb_dim,emb_dim)
            self.out_linear = Linear(emb_dim,emb_dim)
        self.dropout = Dropout(dropout)
    def forward(self,q,k,v,mask=None):
        q = self.q_l(q)
        k = self.k_l(k)
        v = self.v_l(v)

        #we use the view to reshape our q,k,v
        q = q.view(0,-1,self.num_heads,self.head_size)
        k = k.view(0,-1,self.num_heads,self.head_size)
        v = v.view(0,-1,self.num_heads,self.head_size)

        #do scalar dot product
        attention = torch.matmul(q,k.transpose(-2,-1)/torch.sqrt(self.head_size))
        if mask is not None:
            attention = attention.masked_fill(mask==0,float('-inf'))
        attention_weight = Softmax(attention,dim=-1)
        attention_weight = self.dropout(attention,dim=-1)
        attention_output = torch.matmul(attention_weight,v)
        #Concatinate the heads altogther
        attention_output = attention_output.transpose(1,2).contiguous().view(0,-1,self.emb_dim)
        self.out = self.out_linear(attention_output)
        return self.out, attention_weight
    

#FFN(x) = max(0, xW1 + b1)W2 + b2
class PositionWiseForward(nn.Module):
    def __init__(self,input,output_size,dropout=0.1):
        super().__init__()
        self.linear_1 = Linear(input,output_size)
        self.linear_2 = Linear(input,output_size)
        self.dropout = Dropout(dropout)
    def forward(self,x):
        x = ReLU(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
