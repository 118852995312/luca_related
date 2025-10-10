import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..attention import RotateMultiHeadAttention,MultiHeadAttention
from ..Norm import RMSNorm,LayerNorm
from ..utils import FeedForward
from typing import List, Optional, Tuple, Union


class TransformerBlock(nn.Module):
    def __init__(self,config):
        super(TransformerBlock, self).__init__()
        self.layernorm1 = LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.layernorm2 = LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.feedforward = FeedForward(config.hidden_size,4 * config.hidden_size,config.feedforward_dropout_prob)



    def forward(self,inputs:torch.Tensor,
                attention_mask:Optional[torch.FloatTensor] = None)->torch.Tensor:
        residual = inputs
        x = self.attention(inputs,attention_mask)
        x = residual + self.dropout1(x)
        x = self.layernorm1(x)
        residual = x
        x =  self.feedforward(x)
        x = residual + x
        x = self.layernorm2(x)
        return x




class TransformerRotateBlock(nn.Module):
    def __init__(self, config):
        super(TransformerRotateBlock, self).__init__()
        self.rmsnorm1 = RMSNorm(config.hidden_size)
        self.attention = RotateMultiHeadAttention(config)
        self.rmsnorm2 = RMSNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.feedforward = FeedForward(config.hidden_size, 4 * config.hidden_size,config.feedforward_dropout_prob)

    def forward(self, inputs:torch.Tensor,
                freq_cis:torch.Tensor,
                mask:Optional[torch.FloatTensor] = None):
        residual = inputs
        x = self.rmsnorm1(inputs)
        x = self.attention(x,freq_cis,mask)
        x = residual + self.dropout1(x)
        residual = x
        x = self.rmsnorm2(x)
        x = self.feedforward(x)
        x = residual + x
        return x







