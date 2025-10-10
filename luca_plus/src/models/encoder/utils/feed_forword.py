import torch
import torch.nn as nn
import torch.nn.functional as F
from .gelu import GELU


class FeedForward(nn.Module):
    def __init__(self,input_dim,inter_dim,dropout = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim,inter_dim)
        self.linear2 = nn.Linear(inter_dim,input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()


    def forward(self,input):
        return self.linear2(self.dropout(self.activation(self.linear1(input))))
