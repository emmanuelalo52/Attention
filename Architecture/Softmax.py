import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __call__(self,x):
        x = torch.max(x,keepdim=True)
        self.out = torch.exp(x)
        self.out /= sum(self.out)
        return self.out
    def parameter(self):
        return []