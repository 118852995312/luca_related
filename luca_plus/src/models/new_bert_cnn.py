import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from encoder import RotateTransformer_erineembedding,RotateTransformer
from decoder import TextClassifier
from activation import create_activate



class NewBertCNN(nn.Module):
    def __init__(self,config,args):
        super(NewBertCNN, self).__init__()
        self.num_labels = config.num_labels
        self.input_type = config.input_type
        self.has_seq_encoder = False
        self.has_embedding_encoder = False
        if "seq" in self.input_type:
            self.has_seq_encoder = True

        if "matrix" in self.input_type:
            self.has_embedding_encoder = True
        if self.has_seq_encoder:
            self.seq_encoder = RotateTransformer_erineembedding(config)
            self.seq_pooler = TextClassifier(config)
            for idx in range(len(config.seq_fc_size)):
                linear = nn.Linear(config.hidden_size, config.seq_fc_size[idx])
                self.seq_linear.append(linear)
                self.seq_linear.append(create_activate(config.activate_func))
                input_size = config.seq_fc_size[idx]
            self.seq_linear = nn.ModuleList(self.seq_linear)

        if self.has_embedding_encoder:
            self.embedding_pooler = TextClassifier(config)
            input_size = config.embedding_input_size
            for idx in range(len(config.embedding_fc_size)):
                linear = nn.Linear(input_size, config.embedding_fc_size[idx])
                self.embedding_linear.append(linear)
                self.embedding_linear.append(create_activate(config.activate_func))
                input_size = config.embedding_fc_size[idx]
            self.embedding_linear = nn.ModuleList(self.embedding_linear)

        output_size = 0
        if self.has_seq_encoder and self.has_embedding_encoder:
            if hasattr(config, "seq_weight") and hasattr(config, "embedding_weight"):
                self.seq_weight = config.seq_weight
                self.embedding_weight = config.embedding_weight
            else:
                self.seq_weight = None
                self.embedding_weight = None
            self.struct_weight = None
            assert self.seq_weight is None or self.seq_weight + self.embedding_weight == 1.0
            if self.seq_weight is None:  # concat
                output_size = config.seq_fc_size[-1] + config.embedding_fc_size[-1]
            else:  # add
                assert config.seq_fc_size[-1] == config.embedding_fc_size[-1]
                output_size = config.seq_fc_size[-1]

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_mode = args.output_mode
        if args and args.sigmoid:
            if args.output_mode in ["binary_class", "binary-class"]:
                self.classifier = nn.Linear(output_size, 1)
            else:
                self.classifier = nn.Linear(output_size, config.num_labels)
            self.output = nn.Sigmoid()
        else:
            self.classifier = nn.Linear(output_size, config.num_labels)
            if self.num_labels > 1:
                self.output = nn.Softmax(dim=-1)
            else:
                self.output = None

    def forward(
            self,
            input_ids=None,
            seq_attention_masks=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            vectors=None,
            matrices=None,
            matrix_attention_masks=None,
            labels=None,
            task_type_ids = None,
    ):
        seq_pooled_output = None
        if self.has_seq_encoder:
            if input_ids is not None:
                seq_outputs = self.seq_encoder(
                    input_ids,
                    token_type_ids,
                    position_ids,
                    inputs_embeds,
                    task_type_ids,
                )
                if self.seq_pooler is None:
                    seq_pooled_output = seq_outputs
                else:
                    seq_pooled_output = self.seq_pooler(seq_outputs)

                seq_pooled_output = self.dropout(seq_pooled_output)
                for i, layer_module in enumerate(self.seq_linear):
                    seq_pooled_output = layer_module(seq_pooled_output)

        embedding_pooled_output = None
        if self.has_embedding_encoder:
            if self.embedding_pooler:
                embedding_pooled_output = self.embedding_pooler(matrices, mask=matrix_attention_masks)
            else:
                embedding_pooled_output = vectors

            embedding_pooled_output = self.dropout(embedding_pooled_output)
            for i, layer_module in enumerate(self.embedding_linear):
                embedding_pooled_output = layer_module(embedding_pooled_output)

        assert seq_pooled_output is not None or embedding_pooled_output is not None


        if embedding_pooled_output is None:
            pooled_output = seq_pooled_output
        elif seq_pooled_output is None:
            pooled_output = embedding_pooled_output
        else:
            if self.seq_weight is not None and self.embedding_weight is not None:
                pooled_output = torch.add(self.seq_weight * seq_pooled_output,
                                          self.embedding_weight * embedding_pooled_output)
            elif seq_pooled_output is not None and embedding_pooled_output is not None:
                pooled_output = torch.cat([seq_pooled_output, embedding_pooled_output], dim=-1)
            else:
                raise Exception("Not support this type.")

        logits = self.classifier(pooled_output)
        if self.output:
            output = self.output(logits)
        else:
            output = logits
        outputs = [logits, output]

        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception("Not support this output_mode=%s" % self.output_mode)
            outputs = [loss, *outputs]

        return outputs