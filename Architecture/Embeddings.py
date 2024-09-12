import torch
class Embedding:
    def __init__(self,num_embedding,embedding_dim):
        self.weight = torch.randn(num_embedding,embedding_dim)
    def __call__(self,x):
        self.out = self.weight[x]
        return  self.out
    def parameter(self):
        return [self.weight]