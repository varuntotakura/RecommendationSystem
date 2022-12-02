import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMCell(nn.Module):
    '''
        LSTM Cell of the Long Short Term Model
    '''
    def __init__(self, n_tokens, hidden_size, n_tokens_per_class = None):
        super(LSTMCell, self).__init__()
        # Parameters
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size
        self.n_tokens_per_class = n_tokens_per_class
        self.n_classes = int(np.ceil(self.n_tokens * 1. / self.n_tokens_per_class))
        self.n_tokens_actual = self.n_classes * self.n_tokens_per_class
        # Declare weights and biases
        # nn.Parameter to make the variable to work when they are called form the forward function with input parameter
        self.W1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.n_classes), requires_grad=True)
        self.b1 = nn.Parameter(torch.FloatTensor(self.n_classes), requires_grad=True)
        self.W2 = nn.Parameter(torch.FloatTensor(self.n_tokens_per_class, self.hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.FloatTensor(self.n_classes), requires_grad=True)
        self.reset_weights()

    def reset_weights(self):
        '''
            Randomly Initialize weights to start the model and to reset the model during training
        '''
        initrange = 0.1
        self.W1.data.uniform_(-initrange, initrange)
        self.b1.data.fill_(0)
        self.W2.data.uniform_(-initrange, initrange)
        self.b2.data.fill_(0)

    def forward(self, inputs):
        '''
            Compute the layers using the given input in the forward method, to feed the network forward
        '''
        labels = torch.arange(self.n_tokens_actual)
        batch_size, d = inputs.size()
        label_W1 = (labels / self.n_tokens_per_class).long()
        label_W2 = (labels % self.n_tokens_per_class).long()
        label_b1 = torch.matmul(inputs, self.W1) + self.b1
        multi_bias = self.b2[label_W2].repeat(batch_size,1)
        label_b2 = torch.matmul(inputs,self.W2[label_W2].T) + multi_bias
        label_b1 = label_b1.repeat_interleave(self.n_tokens_per_class,dim=1)
        output = torch.add(label_b1, label_b2)
        return output

class LSTMModel(nn.Module):
    '''
        Main Architectiure of the Long Short Term Model
    '''
    def __init__(self, article_shape, n_classes):
        '''
            Call the LSTM Cell and the required layers to the LSTM Model
        '''
        super(LSTMModel, self).__init__()
        self.num_layers = 3
        self.n_classes = n_classes
        self.embedding = nn.Embedding(article_shape[0], embedding_dim=article_shape[1])
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LSTMCell(self.n_classes, 512, n_tokens_per_class = 20))
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.n_classes, 512, n_tokens_per_class = 20))
        
    def forward(self, inputs):
        input_size = inputs[0]
        x = self.embedding(input_size)
        x = F.normalize(x, dim=2)
        x, _ = x.max(axis=1)
        # Adding the LSTM Cell Layers to the Model
        output = self.rnn_cell_list[0](x)
        # Iterate the num of layer times to add the layers to the model
        for i in range(1, self.num_layers):
            output = self.rnn_cell_list[i](x)
        # Formating the shape of the output
        output = output[:, :self.n_classes]
        return output
