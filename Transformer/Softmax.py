import torch
class Softmax:
    def __call__(self,x):
        x = torch.max(x,keepdim=True)
        self.out = torch.exp(x)
        self.out /= sum(self.out)
        return self.out
    def parameter(self):
        return []