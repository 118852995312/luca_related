import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)



def apply_rotary_emb(xq,xk,freq_cis):
    ##先把xq和xk转化成复数表达方式
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1],-1,2))
    freqs_cis = reshape_for_broadcast(freq_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super(MultiHeadAttention, self).__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.linear_q = nn.Linear(config.hidden_size,config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self,x:torch.Tensor)->torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0,2,1,3)


    def forward(self,
                hidden_states:torch.Tensor,
                attention_mask:Optional[torch.FloatTensor] = None,
                )->torch.Tensor:
        b = hidden_states.shape[0]
        query,key,value = self.transpose_for_scores(self.linear_q(hidden_states)),\
                          self.transpose_for_scores(self.linear_k(hidden_states)),\
                          self.transpose_for_scores(self.linear_v(hidden_states))
        scores = torch.matmul(query,key.transpose(-1,-2))/math.sqrt(self.embed_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0,-1e9)
        scores = F.softmax(scores,dim = -1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores,value)
        scores = scores.transpose(1,2).contiguous().reshape(b,-1,self.embed_dim)
        return self.linear_output(scores)


class RotateMultiHeadAttention(nn.Module):
    def __init__(self,config,add_bias_kv=False):
        super(RotateMultiHeadAttention, self).__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.reset_parameters()

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, config.hidden_size))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, config.hidden_size))
        else:
            self.bias_k = self.bias_v = None


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_k.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.linear_v.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.linear_q.weight, gain=nn.init.calculate_gain("relu"))

        nn.init.xavier_uniform_(self.linear_output.weight, gain=nn.init.calculate_gain("relu"))


    def transpose_for_scores(self,x:torch.Tensor)->torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0,2,1,3)


    def forward(self,hidden_states:torch.Tensor,
                freqs_cis:torch.Tensor,
                attention_mask:Optional[torch.FloatTensor] = None
                )->torch.Tensor:
        b,max_seq,_ = hidden_states.shape
        query,key,value = self.transpose_for_scores(self.linear_q(hidden_states)),\
                          self.transpose_for_scores(self.linear_k(hidden_states)),\
                          self.transpose_for_scores(self.linear_v(hidden_states))

        query,key = apply_rotary_emb(query,key,freqs_cis)
        xq = query.transpose(1,2)
        key = key.transpose(1, 2)
        value = value.transpose(1,2)
        scores = torch.matmul(xq,key.transpose(-1,-2))/math.sqrt(self.embed_dim)

        if attention_mask is not None:
            scores =  scores.masked_fill(attention_mask == 0,-1e9)
        scores = F.softmax(scores,dim = -1).type_as(xq)
        scores = self.dropout(scores)
        # scores = self.dropout(scores)
        scores = torch.matmul(scores,value)
        scores = scores.transpose(1,2).contiguous().reshape(b,-1,self.embed_dim)
        return self.linear_output(scores)

