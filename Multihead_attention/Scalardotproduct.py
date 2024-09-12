import torch
from ..Transformer.Softmax import Softmax
from Architecture.dropout import Dropout
class ScalarDotProduct:
    def __init__(self,temperature,attn_dropout=0.1):
        self.temperature = temperature
        self.dropout = Dropout(self.dropout)
    def __call__(self,q,k,v,mask=None):
        attention = torch.matmul(q/self.temperature,k.transpose(2,3))
        if mask is not None:
            attention = attention.masked_fill(mask==0,-1e9)
        attention = self.dropout(Softmax(attention,dim=-1))
        self.out = torch.matmul(attention,v)
        return self.out
    def parameters(self):
        return[self.dropout]