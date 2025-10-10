import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union
from ..Norm import RMSNorm


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class BertEmbedding(nn.Module):
    def __init__(self,config):
        super(BertEmbedding, self).__init__()
        if hasattr(config,"no_token_embeddings"):
            self.no_token_embeddings = config.no_token_embeddings
        else:
            self.no_token_embeddings = False

        if not self.no_token_embeddings:
            self.word_embeddings = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=config.pad_token_id)

        if hasattr(config,"no_position_embeddings"):
            self.no_position_embeddings = config.no_position_embeddings
        else:
            self.no_position_embeddings = False

        if not self.no_position_embeddings:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.hidden_size)


        if hasattr(config,"no_token_type_embeddings"):
            self.no_token_type_embeddings = config.no_token_type_embeddings
        else:
            self.no_token_type_embeddings = False

        if not self.no_token_type_embeddings:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size,config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if not self.no_position_embeddings:
            self.position_embeddings = getattr(config,"position_embeddding_type","absolute")
            self.register_buffer("position_ids",torch.arange(config.max_position_embeddings).expand((1,-1)))
        if not self.no_token_type_embeddings and not self.no_position_embeddings:
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(),dtype=torch.long),
                persistent=False,
            )

    def forward(self,
                input_ids:Optional[torch.LongTensor] = None,
                token_type_ids:Optional[torch.LongTensor] = None,
                position_ids:Optional[torch.LongTensor] = None,
                inputs_embeds:Optional[torch.LongTensor] = None,
                past_key_values_length: int = 0,
                ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]


        if not self.no_position_embeddings and position_ids is None:
            position_ids = self.position_ids[:,past_key_values_length:past_key_values_length + seq_length]

        if not self.no_token_type_embeddings and token_type_ids is None:
            if hasattr(self,"token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:,:seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0],seq_length)
                token_type_ids = buffered_token_type_ids_expanded

            else:
                token_type_ids = torch.zeros(input_shape,dtype = torch.long,device = input_ids.device if input_ids is not None else inputs_embeds.device)


        if self.no_token_embeddings and inputs_embeds is None:
            raise Exception("The model has not token_embeddings layer, the inputs_embeds cannot None")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        if not self.no_token_type_embeddings:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings


        if not self.no_position_embeddings and self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class ErineEmbedding(nn.Module):
    def __init__(self,config):
        super(ErineEmbedding, self).__init__()
        if hasattr(config,"no_token_embeddings"):
            self.no_token_embeddings = config.no_token_embeddings
        else:
            self.no_token_embeddings = False

        if not self.no_token_embeddings:
            self.word_embeddings = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=config.pad_token_id)

        if hasattr(config,"no_position_embeddings"):
            self.no_position_embeddings = config.no_position_embeddings
        else:
            self.no_position_embeddings = False

        if not self.no_position_embeddings:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.hidden_size)


        if hasattr(config,"no_token_type_embeddings"):
            self.no_token_type_embeddings = config.no_token_type_embeddings
        else:
            self.no_token_type_embeddings = False

        if not self.no_token_type_embeddings:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size,config.hidden_size)


        if hasattr(config,"no_task_type_embeddings"):
            self.no_task_type_embeddings = config.no_task_type_embeddings
        else:
            self.no_task_type_embeddings = False


        if not self.no_task_type_embeddings:
            self.task_type_embeddings = nn.Embedding(config.task_vocab_size,config.hidden_size)



        self.RMSNorm = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if not self.no_position_embeddings:
            self.position_embeddings = getattr(config,"position_embeddding_type","absolute")
            self.register_buffer("position_ids",torch.arange(config.max_position_embeddings).expand((1,-1)))
        if not self.no_token_type_embeddings and not self.no_position_embeddings:
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(),dtype=torch.long),
                persistent=False,
            )

        if not self.no_task_type_embeddings and not self.no_position_embeddings:
            self.register_buffer(
                "task_type_ids",
                torch.zeros(self.position_ids.size(),dtype = torch.long),
                persistent=False,
            )

    def forward(self,
                input_ids:Optional[torch.LongTensor] = None,
                token_type_ids:Optional[torch.LongTensor] = None,
                position_ids:Optional[torch.LongTensor] = None,
                inputs_embeds:Optional[torch.LongTensor] = None,
                task_type_ids:Optional[torch.LongTensor] = None,
                past_key_values_length: int = 0,
                ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]


        if not self.no_position_embeddings and position_ids is None:
            position_ids = self.position_ids[:,past_key_values_length:past_key_values_length + seq_length]

        if not self.no_token_type_embeddings and token_type_ids is None:
            if hasattr(self,"token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:,:seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0],seq_length)
                token_type_ids = buffered_token_type_ids_expanded

            else:
                token_type_ids = torch.zeros(input_shape,dtype = torch.long,device = input_ids.device if input_ids is not None else inputs_embeds.device)


        if not self.no_task_type_embeddings and task_type_ids is None:
            buffered_task_type_ids = self.task_type_ids[:, :seq_length]
            buffered_task_type_ids_expanded = buffered_task_type_ids.expand(input_shape[0], seq_length)
            task_type_ids = buffered_task_type_ids_expanded

        if self.no_token_embeddings and inputs_embeds is None:
            raise Exception("The model has not token_embeddings layer, the inputs_embeds cannot None")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        if not self.no_token_type_embeddings:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings


        if not self.no_task_type_embeddings:
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings  += task_type_embeddings


        if not self.no_position_embeddings and self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.RMSNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
