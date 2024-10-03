from dependencies import *
class TanhGelu(nn.Module):
    def forward(self,input):
        return (0.5 * input) * (1.0 + torch.tanh(math.sqrt(2.0/math.pi)*(input + 0.044715 * torch.pow(input,3))))