

import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-9):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.ones(hidden_size))



    def forward(self,x):
        var = x.var(dim = -1,keepdim = True)
        x_norm = x / torch.sqrt(var + self.eps)
        out = self.beta * x_norm
        return out