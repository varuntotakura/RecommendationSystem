import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None):
        super(HierarchicalSoftmax, self).__init__()
        # Parameters
        self.ntokens = ntokens#the number of ouput.(n_classes)
        self.nhid = nhid#dimension: the same length of customer dimension.(512)
        self.ntokens_per_class = ntokens_per_class#how many children one intermidiate node.(20)
        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))#intermidiate nodes.(3630)
        self.ntokens_actual = self.nclasses * self.ntokens_per_class#72600
        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)
        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.ntokens_per_class, self.nhid), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)

    def forward(self, inputs):
        labels = torch.arange(self.ntokens_actual)###72600 
        batch_size, d = inputs.size()
        label_position_top = (labels / self.ntokens_per_class).long()#which position is the top layer.###[0,0,..,0,....,3659,3659]
        label_position_bottom = (labels % self.ntokens_per_class).long()#which position is the bottom layer.###[0,1,2,..,19,1,2,...,19,..]
        layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b###[256, 3630]
        multi_bias = self.layer_bottom_b[label_position_bottom].repeat(batch_size,1)###[256,72600]
        layer_bottom_logits = torch.matmul(inputs,self.layer_bottom_W[label_position_bottom].T) + multi_bias###[256,72600]
        layer_top_logits = layer_top_logits.repeat_interleave(self.ntokens_per_class,dim=1)###[256,72600]#match the top classes and the bottom classes.
        target_logits = torch.add(layer_top_logits,layer_bottom_logits)#get the final logits

        return target_logits

class HMModel(nn.Module):
    def __init__(self, article_shape, n_classes):
        super(HMModel, self).__init__()
        self.n_classes = n_classes
        self.article_emb = nn.Embedding(article_shape[0], embedding_dim=article_shape[1])
        self.hier = HierarchicalSoftmax(self.n_classes, 512, ntokens_per_class = 20)
        
    def forward(self, inputs):
        article_hist, week_hist = inputs[0], inputs[1]
        x = self.article_emb(article_hist)
        x = F.normalize(x, dim=2)
        x, indices = x.max(axis=1)
        logits = self.hier(x)
        logits = logits[:, :self.n_classes]

        return logits