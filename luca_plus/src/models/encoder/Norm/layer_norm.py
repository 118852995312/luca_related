
import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-9):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.ones(hidden_size))
        self.gamma = nn.Parameter(torch.zeros(hidden_size))


    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        var = x.var(dim = -1,keepdim = True)
        x_norm = (x - mean)/ torch.sqrt(var + self.eps)
        out = self.beta * x_norm + self.gamma
        return out



