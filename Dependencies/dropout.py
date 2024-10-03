import torch
class Dropout:
    def __init__(self,training,p=0.05):
        self.training = training
        self.p = p
    def __call__(self,x):
        if self.training:
            x = torch.randn([0.0,1.0])
            x *= x/(1-self.p)
            self.out = x
        return self.out