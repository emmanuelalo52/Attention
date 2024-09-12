import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,fan_in,fan_out,bias=True):
        super().__init__()
        self.weight = torch.randn(fan_in, fan_out)/ fan_out * 0.5
        self.bias = torch.zeros(fan_out) if bias else None
    def forward(self,x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out