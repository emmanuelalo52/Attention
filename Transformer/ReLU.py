import torch
class ReLU:
    def __call__(self, x):
        x = torch.tensor(x,dtype=torch.float)
        self.out = torch.max(torch.tensor(0.0),x)
        return self.out