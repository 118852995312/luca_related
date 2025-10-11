import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..Norm import RMSNorm
from .TransformerBlock import TransformerBlock,TransformerRotateBlock
from ..embeddings import BertEmbedding,ErineEmbedding
from ..utils import precompute_freqs_cis
from typing import List, Optional, Tuple, Union

class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer, self).__init__()
        self.embedding = BertEmbedding(config)
        self.transformerList = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])



    def forward(self,inputs:torch.Tensor,
                attention_mask:Optional[torch.FloatTensor] = None)->torch.Tensor:
        x = self.embedding(inputs)
        mask = attention_mask[:, None, None, :]
        for transformer in self.transformerList:
            x = transformer.forward(x, mask)

        return x


class RotateTransformer(nn.Module):
    def __init__(self, config):
        super(RotateTransformer, self).__init__()
        self.embedding = BertEmbedding(config)
        self.norm = RMSNorm(config.hidden_size)
        self.freq_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_seq_len)
        self.transformerList = nn.ModuleList([TransformerRotateBlock(config) for _ in range(config.num_layers)])


    def forward(self,
                inputs:torch.Tensor,
                attention_mask:Optional[torch.FloatTensor] = None)->torch.Tensor:
        x = self.embedding(inputs)
        self.freq_cis = self.freq_cis.to(x.device)

        mask = attention_mask[:,None,None,:]


        for transformer in self.transformerList:
            x = transformer.forward(x, mask,self.freq_cis)
        x = self.norm(x)
        return x





class RotateTransformer_erineembedding(nn.Module):
    def __init__(self,config):
        super(RotateTransformer_erineembedding, self).__init__()
        self.embedding = ErineEmbedding(config)
        self.norm = RMSNorm(config.hidden_size)
        self.freq_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_seq_len)
        self.transformerList = nn.ModuleList([TransformerRotateBlock(config) for _ in range(config.num_layers)])
        self.device = config.device


    def forward(self,
                inputs:torch.Tensor,
                attention_mask:Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids:Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.LongTensor] = None,
                task_type_ids: Optional[torch.LongTensor] = None,
                )->torch.Tensor:
        x = self.embedding(input_ids = inputs,
                           token_type_ids = token_type_ids,
                           position_ids = position_ids,
                           inputs_embeds = inputs_embeds,
                           task_type_ids = task_type_ids
                           ).to(self.device)
        self.freq_cis = self.freq_cis.to(x.device)

        mask = attention_mask[:,None,None,:]


        for transformer in self.transformerList:
            x = transformer.forward(x,self.freq_cis, mask)
        x = self.norm(x)
        return x

