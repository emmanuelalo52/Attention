from dependencies import *
class Softmax(nn.Module):
    super().__init__()
    def forward(self, x):
        x_max = torch.max(x, dim=-1, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    
    def parameters(self):
        return []