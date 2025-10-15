


import torch
import torch.nn as nn
import torch.nn.functional as F





class TextCNN(nn.Module):
    def __init__(self,config):
        super(TextCNN, self).__init__()
        self.convs1 = nn.Conv1d(in_channels=config.hidden_size,out_channels=config.hidden_size,kernel_size=3,stride=1)
        self.convs2 = nn.Conv1d(in_channels=config.hidden_size,out_channels=config.hidden_size,kernel_size=4,stride=1)
        self.convs3 = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=5,
                                stride=1)


        self.pool1d = nn.MaxPool1d(kernel_size=config.max_seq_len - 1)
        self.pool2d = nn.MaxPool1d(kernel_size=config.max_seq_len - 2)
        self.pool3d = nn.MaxPool1d(kernel_size=config.max_seq_len - 3)
        self.dropout = nn.Dropout(config.cnn_dropout)


    def forward(self,embedded:torch.Tensor)->torch.Tensor:
        embedded = embedded.transpose(1,2)
        batch,embed_dim,seq_len = embedded.shape
        Conv_output = []
        if embedded.shape[-1] < 5:
            target_shape = (batch,embed_dim,5)
            target_embedded = torch.zeros(target_shape)
            target_embedded[:batch,:embed_dim,:seq_len] = embedded
            embedded = target_embedded

        conv_output1 = self.convs1(embedded)
        conv_output1 = self.pool1d(conv_output1)
        conv_output1 = torch.squeeze(conv_output1,dim = -1)
        Conv_output.append(conv_output1)

        conv_output2 = self.convs2(embedded)
        conv_output2 = self.pool2d(conv_output2)
        conv_output2 = torch.squeeze(conv_output2, dim=-1)
        Conv_output.append(conv_output2)

        conv_output3 = self.convs3(embedded)
        conv_output3 = self.pool3d(conv_output3)
        conv_output3 = torch.squeeze(conv_output3, dim=-1)
        Conv_output.append(conv_output3)

        concat_output = torch.cat(Conv_output, dim=1)  ##拼接所有卷积核的输出
        # 全连接层
        concat_output = self.dropout(concat_output)  ## batch_size * 3 * embed_dim
        # logits = self.fc(concat_output)  # (batch_size, num_classes)
        return concat_output









class TextClassifier(nn.Module):
    def __init__(self,config):
        super(TextClassifier,self).__init__()
        self.textcnn = TextCNN(config)
        self.fc = nn.Linear(config.hidden_size * 4, config.num_labels)
        self.dropout = nn.Dropout(config.cnn_dropout)

        # self.classifier = nn.Linear(embed_dim, num_class)


    def forward(self,context_outputs):
        # context_outputs = self.encoder(input_ids = input_ids,token_type_ids = token_type_ids)

        ##处理第一个维度
        context_outputs = context_outputs[0]
        pooled_output = self.dropout(context_outputs[:,0,:])
        # logits = self.classifier(pooled_output)

        ##处理剩余维度

        cnn_features = self.textcnn(embedded = context_outputs[:,1:,:])

        all_features = torch.cat([pooled_output,cnn_features],dim=1)

        logits = F.softmax(self.fc(all_features))
        return logits
