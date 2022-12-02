import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HiddenLayers(nn.Module):
    '''
        Hidden Layers of the Hidden Markov Model
    '''
    def __init__(self, n_tokens, hidden_size, n_tokens_per_class = None):
        super(HiddenLayers, self).__init__()
        # Inititalize Parameters
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size
        self.n_tokens_per_class = n_tokens_per_class
        self.n_classes = int(np.ceil(self.n_tokens * 1. / self.n_tokens_per_class))
        self.n_tokens_actual = self.n_classes * self.n_tokens_per_class
        # Declare the component of the Neural Network, i.e., weights and biases
        # Using nn.Parameter to make the variable to work when they are called form the forward function with input parameter
        self.W1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.n_classes), requires_grad=True)
        self.b1 = nn.Parameter(torch.FloatTensor(self.n_classes), requires_grad=True)
        self.W2 = nn.Parameter(torch.FloatTensor(self.n_tokens_per_class, self.hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.FloatTensor(self.n_classes), requires_grad=True)
        self.reset_weights()

    def reset_weights(self):
        # Randomly Initialize weights to start the model and to reset the model during training
        initrange = 0.1
        self.W1.data.uniform_(-initrange, initrange)
        self.b1.data.fill_(0)
        self.W2.data.uniform_(-initrange, initrange)
        self.b2.data.fill_(0)

    def forward(self, inputs):
        # Forward is the main component of the PyTorch Neural Network Model, where all the computations on the input data is performed
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

class HMModel(nn.Module):
    '''
        Main Architectiure of the Hidden Markov Model
    '''
    def __init__(self, article_shape, n_classes):
        # Component of the HMM Model and calling the Hidden Layers
        super(HMModel, self).__init__()
        self.n_classes = n_classes
        self.embedding = nn.Embedding(article_shape[0], embedding_dim=article_shape[1])
        self.hidden = HiddenLayers(self.n_classes, 512, n_tokens_per_class = 20)
        
    def forward(self, inputs):
        input_size = inputs[0]
        x = self.embedding(input_size)
        x = F.normalize(x, dim=2)
        x, _ = x.max(axis=1)
        output = self.hidden(x)
        # Formating the shape of the output
        output = output[:, :self.n_classes]
        return output