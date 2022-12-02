import math
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
beta = 0.5
scale = 100

class Environment:
    '''
        Reinforcement Learning Requires an Environment where it could run and train itself with the help of rewards generated for each action it has performed
        Similarly, in this dataset, we are using the articles, customers, and transactions as the variables inside the environment
        Initiating and calculation of the reward is performed
    '''
    def __init__(self, transactions, emb_customers, emb_articles, df_customers, df_articles):
        self.transactions = transactions
        self.emb_customers = emb_customers
        self.emb_prod = emb_articles
        self.customers = df_customers
        self.articles = df_articles
        # self.reset()
        
    def reset(self):
        '''
            Reset is when a new environment is created
        '''
        self.done = False
        self.current_pos = 0
        self.customer_id = self.transactions.iloc[self.current_pos, 0]
        self.customer_index = self.customers.index[self.customers['customer_id']==self.customer_id].tolist()[0]
        self.customer_embed = self.emb_customers(torch.LongTensor([self.customer_index]).to(device, dtype=torch.long))
        ###
        # {outcome (for debugging)}
        #   customer_id: 00039306476aaf41a07fed942884f16b30abfa83a2a8bea972019098d6406793 
        #   customer_Index: 80 
        #   emb_customers = Embedding(1371980, 30)
        #   customer_embed = tensor([[ 0.5234,  1.1311,  0.9114, -1.6386, -0.1985,  1.3215, -0.2250,  0.6025,
        #     0.3260,  0.8365,  1.2712,  0.1888,  0.2963, -0.7545,  0.9620,  1.8377,
        #    -0.1847,  0.9194,  0.7094,  1.5421,  2.0885,  0.7273, -2.0009,  0.5741,
        #    -1.9650, -0.0630,  0.4846, -1.0385, -0.1492, -1.1125]],
        #   device='cuda:0', grad_fn=<EmbeddingBackward0>)
        ###
        return self.customer_embed
    
    def step(self, act):
        '''
            Step is the place where the environment needs to adjust its values as per the actions performed
            ###
            #   customer_id: 00039306476aaf41a07fed942884f16b30abfa83a2a8bea972019098d6406793 
            #   customer_Index: 80 
            #   emb_customers = Embedding(1371980, 30)
            #   customer_embed = tensor([[ 0.5234,  1.1311,  0.9114, -1.6386, -0.1985,  1.3215, -0.2250,  0.6025,
            #     0.3260,  0.8365,  1.2712,  0.1888,  0.2963, -0.7545,  0.9620,  1.8377,
            #    -0.1847,  0.9194,  0.7094,  1.5421,  2.0885,  0.7273, -2.0009,  0.5741,
            #    -1.9650, -0.0630,  0.4846, -1.0385, -0.1492, -1.1125]],
            #   device='cuda:0', grad_fn=<EmbeddingBackward0>)
            ###
        '''
        reward = 0
        self.product_id = self.transactions.iloc[self.current_pos, 1]
        self.prod_index = self.articles.index[self.articles['article_id']==self.product_id].tolist()[0]
        reward = self.compute_reward(act, self.prod_index)
        if (self.current_pos < len(self.transactions.index)):
            self.current_pos += 1
            if (self.current_pos < len(self.transactions.index)):
                self.customer_id = self.transactions.iloc[self.current_pos, 0]
                self.customer_index = self.customers.index[self.customers['customer_id']==self.customer_id].tolist()[0]
                self.customer_embed = self.emb_customers(torch.LongTensor([self.customer_index]).to(device, dtype=torch.long))
                self.done = False
                return self.customer_embed, reward, self.done
            else:
                self.done = True

                return self.customer_embed, reward, self.done
    
    def compute_reward(self, proposed_articles, actual_product):
        '''
            Computation of the Reward is performed with the help of actions that were taken towards reaching the goal
            The reward will be lo if the actions move far away form the goal
            And the reward will be high if the actions are performed towards achieving the goal
        '''
        if (actual_product in proposed_articles):
            position = proposed_articles.index(actual_product)
            proposed_articles = proposed_articles[:position+1]

        embed_proposed = self.emb_prod[torch.LongTensor(proposed_articles).to(device, dtype=torch.long)]
        embed_actual = self.emb_prod[torch.LongTensor([actual_product]).to(device, dtype=torch.long)]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(embed_proposed, embed_actual)
        coeffs = [math.pow(beta, t) for t in range(len(proposed_articles))]
        coeffs = torch.Tensor(coeffs).to(device, dtype=torch.float)
        output = torch.mul(output, coeffs)
        reward = torch.sum(output)
        reward = torch.mul(reward, scale)
        reward = reward.item()
        return reward

class Actor(torch.nn.Module):
    '''
        In Actor-Critic Model, Actor works as an agent by which all the possible actions are performed
    '''
    def __init__(self, n_input, n_weight_out, n_features_out, product_feats):
        super(Actor, self).__init__()
        self.n_input = n_input
        self.n_weight_out = n_weight_out
        self.n_features_out = n_features_out
        self.feats = product_feats
        l1 = 4 * self.n_input
        l2 = 2 * l1
        self.l1 = nn.Linear(self.n_input, l1)
        self.l2 = nn.Linear(l1, l2)
        self.l3 = nn.Linear(l2, self.n_weight_out * self.n_features_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        the_weights = x.view(-1, self.n_weight_out, self.n_features_out)
        product_trans = torch.transpose(self.feats, 0, 1)
        all_scalar_prods = torch.matmul(the_weights, product_trans)
        argmax = torch.argmax(all_scalar_prods, dim=2)

        return argmax

class Critic(torch.nn.Module):
    '''
        In Actor-Critic Model, Critic works as an agent which negates to the actions performed by Actor and try to improve the model
    '''
    def __init__(self, n_input, n_rec_prod):
        # n_input: number of features that represents a customer
        # n_rec_prod: number of articles recommend by our actor
        super(Critic, self).__init__()
        self.n_input = n_input
        self.n_rec_prod = n_rec_prod
        l1 = 4*(self.n_input + self.n_rec_prod)
        l2 = l1
        l3 = 1
        self.l1 = nn.Linear(self.n_input + self.n_rec_prod, l1)
        self.l2 = nn.Linear(l1, l2)
        self.l3 = nn.Linear(l2, l3)

    def forward(self, x, act):
        x = torch.cat((x, act), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Model(torch.nn.Module):
    '''
        Main Model which combines both Actor and Critic Models and make it a complete model
    '''
    def __init__(self, actor, critic):
        super(Model, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        act = self.actor(x)
        crit = self.critic(x, act)
        return act, crit