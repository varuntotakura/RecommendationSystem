import math
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
beta = 0.5
scale = 100

class Environment:
    def __init__(self, data_transactions, emb_customers, emb_products, df_customers, df_products):
        self.data = data_transactions
        self.emb_customers = emb_customers
        self.emb_prod = emb_products
        self.customers = df_customers
        self.products = df_products
        # self.reset()
        
    def reset(self):
        self.done = False
        self.current_pos = 0
        # we must return the informations about the current user (and the products bought by him)
        # let's go first with the current user by getting his id
        self.userid = self.data.iloc[self.current_pos, 0]
        # from the userid, we must get its corresponding index  in df_customers
        self.user_index = self.customers.index[self.customers['customer_id']==self.userid].tolist()[0]
        # now we can get the user embedding informations from self.emb_customers
        self.user_embed = self.emb_customers(torch.LongTensor([self.user_index]).to(device, dtype=torch.long))
        # print(self.userid, self.user_index, self.emb_customers, self.user_embed)
        # print(torch.argmin(torch.norm(self.user_embed, dim=1)))
        ###
        #   UserID: 00039306476aaf41a07fed942884f16b30abfa83a2a8bea972019098d6406793 
        #   User_Index: 80 
        #   emb_customers = Embedding(1371980, 30)
        #   user_embed = tensor([[ 0.5234,  1.1311,  0.9114, -1.6386, -0.1985,  1.3215, -0.2250,  0.6025,
        #     0.3260,  0.8365,  1.2712,  0.1888,  0.2963, -0.7545,  0.9620,  1.8377,
        #    -0.1847,  0.9194,  0.7094,  1.5421,  2.0885,  0.7273, -2.0009,  0.5741,
        #    -1.9650, -0.0630,  0.4846, -1.0385, -0.1492, -1.1125]],
        #   device='cuda:0', grad_fn=<EmbeddingBackward0>)
        #   tensor(0, device='cuda:0')
        ###
        # in this first version we are not going to consider the last products used by our customer
        return self.user_embed
    
    def step(self, act):
        # here act is a list of 12 (num_prod_to_rec) products to recommend by their indices.
        # For example, it could be [1, 2, 3, ..., 11, 12] 
        reward = 0
        # First, me must find the index of the actual product used by the customer in self.products
        # let's go first with the current product by getting his id
        self.product_id = self.data.iloc[self.current_pos, 1]
        # from the product_id, we must get its corresponding index in df_products
        self.prod_index = self.products.index[self.products['article_id']==self.product_id].tolist()[0]
        reward = self.compute_reward(act, self.prod_index)
        if (self.current_pos < len(self.data.index)):
#             if ((self.current_pos % episode_length)!=0):
            self.current_pos += 1
            if (self.current_pos < len(self.data.index)):
                self.userid = self.data.iloc[self.current_pos, 0]
                self.user_index = self.customers.index[self.customers['customer_id']==self.userid].tolist()[0]
                self.user_embed = self.emb_customers(torch.LongTensor([self.user_index]).to(device, dtype=torch.long))
                self.done = False
                return self.user_embed, reward, self.done
            else:
                self.done = True
                return self.user_embed, reward, self.done
    
    def compute_reward(self, proposed_products, actual_product):
        # print(proposed_products, actual_product)
        if (actual_product in proposed_products):
            # take the position where the matching occurs
            position = proposed_products.index(actual_product)
            proposed_products = proposed_products[:position+1]
        embed_proposed = self.emb_prod[torch.LongTensor(proposed_products).to(device, dtype=torch.long)]
        embed_actual = self.emb_prod[torch.LongTensor([actual_product]).to(device, dtype=torch.long)]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(embed_proposed, embed_actual)
        coeffs = [math.pow(beta, t) for t in range(len(proposed_products))]
        # convert coeffs to tensor
        coeffs = torch.Tensor(coeffs).to(device, dtype=torch.float)
        # apply element wise multiplication to output
        output = torch.mul(output, coeffs)
        # we sum that and multiply by 100
        reward = torch.sum(output)
        reward = torch.mul(reward, scale)
        reward = reward.item()
        return reward

class Actor(torch.nn.Module):
    def __init__(self, n_input, n_weight_out, n_features_out, product_feats):
        # n_input: number of features that represents a customer
        # n_weight_out: number of products we want to recommend represent by their features
        # n_features_out: number of features of each vector of recommend products
        # product_feats: matrix of vector products. Each line as the same number of
        # features as n_features_out
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
        # take the dot product between the_weights and self.feats
        product_trans = torch.transpose(self.feats, 0, 1)
        all_scalar_prods = torch.matmul(the_weights, product_trans)
        # output shape of previous operation : (1, n_weight_out, num_of_products)
        # find the indices with the highest scalar products
        argmax = torch.argmax(all_scalar_prods, dim=2)
        return argmax

class Critic(torch.nn.Module):
    def __init__(self, n_input, n_rec_prod):
        # n_input: number of features that represents a customer
        # n_rec_prod: number of products recommend by our actor
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
    def __init__(self, actor, critic):
        super(Model, self).__init__()
        self.actor = actor
        self.critic = critic
    def forward(self, x):
        act = self.actor(x)
        crit = self.critic(x, act)
        return act, crit