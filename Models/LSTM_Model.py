import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMCell(nn.Module):
    def __init__(self, n_tokens, hidden_size, n_tokens_per_class = None):
        super(LSTMCell, self).__init__()
        # Parameters
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size
        self.n_tokens_per_class = n_tokens_per_class
        self.nclasses = int(np.ceil(self.n_tokens * 1. / self.n_tokens_per_class))
        self.n_tokens_actual = self.nclasses * self.n_tokens_per_class
        self.W1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.nclasses), requires_grad=True)
        self.b1 = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)
        self.W2 = nn.Parameter(torch.FloatTensor(self.n_tokens_per_class, self.hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)
        self.reset_weights()

    def reset_weights(self):
        initrange = 0.1
        self.W1.data.uniform_(-initrange, initrange)
        self.b1.data.fill_(0)
        self.W2.data.uniform_(-initrange, initrange)
        self.b2.data.fill_(0)

    def forward(self, inputs):
        labels = torch.arange(self.n_tokens_actual)
        batch_size, d = inputs.size()
        label_W1 = (labels / self.n_tokens_per_class).long()
        label_W2 = (labels % self.n_tokens_per_class).long()
        label_b1 = torch.matmul(inputs, self.W1) + self.b1
        multi_bias = self.b2[label_W2].repeat(batch_size,1)
        label_b2 = torch.matmul(inputs,self.W2[label_W2].T) + multi_bias
        label_b1 = label_b1.repeat_interleave(self.n_tokens_per_class,dim=1)
        target_logits = torch.add(label_b1, label_b2)

        return target_logits

class LSTMModel(nn.Module):
    def __init__(self, article_shape, n_classes):
        super(LSTMModel, self).__init__()
        self.num_layers = 3
        self.n_classes = n_classes
        self.article_emb = nn.Embedding(article_shape[0], embedding_dim=article_shape[1])
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LSTMCell(self.n_classes, 512, n_tokens_per_class = 20))
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.n_classes, 512, n_tokens_per_class = 20))
        
    def forward(self, inputs):
        article_hist, week_hist = inputs[0], inputs[1]
        x = self.article_emb(article_hist)
        x = F.normalize(x, dim=2)
        x, indices = x.max(axis=1)
        logits = self.rnn_cell_list[0](x)
        for i in range(1, self.num_layers):
            logits = self.rnn_cell_list[i](x)
        logits = logits[:, :self.n_classes]

        return logits