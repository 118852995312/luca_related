import torch
import torch.nn as nn


class GlobalMaskValueAttentionPooling1d(nn.Module):
    def __init__(self,embed_size,units=None,use_additive_bias=False,use_attention_bias=False):
        super(GlobalMaskValueAttentionPooling1d, self).__init__()
        self.embed_size = embed_size
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.units = units
        self.U = nn.Parameter(torch.Tensor(self.embed_size,self.units))
        self.V = nn.Parameter(torch.Tensor(self.embed_size,self.units))

        if use_additive_bias:
            self.b1 = nn.Parameter(torch.Tensor(self.units))
            nn.init.trunc_normal_(self.b1, std=0.01)

        if self.use_attention_bias:
            self.b2 = nn.Parameter(torch.Tensor(self.embed_size))
            nn.init.trunc_normal_(self.b2,std = 0.01)

        self.W = nn.Parameter(torch.Tensor(self.units,self.embed_size))

        nn.init.trunc_normal_(self.U,std = 0.01)
        nn.init.trunc_normal_(self.V, std=0.01)
        nn.init.trunc_normal_(self.W, std=0.01)


    def forward(self,x,mask = None):
        q = torch.matmul(x,self.U)
        k = torch.matmul(x,self.V)
        if self.use_additive_bias:
            h = torch.tanh(q + k + self.b1)
        else:
            h = torch.tanh(q + k)

        if self.use_attention_bias:
            e = torch.matmul(h,self.W) + self.b2
        else:
            e = torch.matmul(h,self.W)

        if mask is not None:
            attention_probs = nn.Softmax(dim = 1)(e + torch.unsqueeze((1.0 - mask)* -10000,dim = -1))
        else:
            attention_probs = nn.Softmax(dim = 1)(e)

        x = torch.sum(attention_probs * x,dim = 1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.embed_size) + ' -> ' + str(self.embed_size) + ')'







