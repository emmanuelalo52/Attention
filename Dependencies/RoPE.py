from dependencies import *
class RoPE(nn.Module):
    def __init__(self,dim,base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    def forward(self,input_pos,x):
        B,T,C = x.shape
        input_pos = torch.arange(T,torch.long,device=x.device)
        sincos = torch.einsum('i,j->ij', input_pos, self.inv_freq)
        sin,cos = torch.sin(sincos), torch.cos(sincos)
        #expand each rotary element through each batch item
        sin = sin.unsqueeze(0).expand(x.shape[0],-1,-1)
        cos = cos.unsqueeze(0).expand(x.shape[0],-1,-1)
        #split the input into 2
        #e^(iθ) * (x + yi) = (x cos θ - y sin θ) + i(x sin θ + y cos θ)
        x1,x2 = x[..., 0::2], x[..., 1::2]
        rope = torch.cat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], dim=-1)
        return rope