from dependencies import *
import Softmax
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
    
    def forward(self, q, k, v):
        dim_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim_k)
        weights = self.softmax(scores)
        return torch.matmul(weights, v)