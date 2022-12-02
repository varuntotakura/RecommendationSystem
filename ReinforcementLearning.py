import numpy as np
import copy
from collections import deque
import random
import torch
import matplotlib.pyplot as plt

from DatasetsPreprocess.RL_Preprocess import RL_Dataset
from Models.ReinforcementLearning_Model import Environment, Actor, Critic, Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
len_embed = 30
num_prod_to_rec = 12 # number of products to recommend to a user at each time step
sync_freq = 50
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
epochs = 1
losses = []
mem_size = 5000
batch_size = 256
replay = deque(maxlen=mem_size)
syn_freq = 500
gamma = 0.9
j=0

df_transactions, df_customers, df_products = RL_Dataset()

# embed_prod = nn.Embedding(len(df_products.index), len_embed)
embed_prod = torch.randn(len(df_products.index), len_embed)
embed_prod = embed_prod.to(device, dtype=torch.float)

date_split = '2020-09-20'
df_transactions = df_transactions[date_split:]
date_split = '2020-09-22'
train = df_transactions[:date_split]
test = df_transactions[date_split:]

embed_custom = torch.nn.Embedding(len(df_customers.index), len_embed)
embed_custom = embed_custom.to(device, dtype=torch.float)

actor_model = Actor(n_input=len_embed, n_weight_out=num_prod_to_rec, n_features_out=30, product_feats=embed_prod)
actor_model.to(device)
critic_model = Critic(n_input=len_embed, n_rec_prod=num_prod_to_rec)
critic_model.to(device)
model = Model(actor_model, critic_model)
model.to(device)
MainModel = copy.deepcopy(model)
MainModel.to(device)
MainModel.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(epochs):
    # start a new environment
    env = Environment(df_transactions, embed_custom, embed_prod, df_customers, df_products)
    state1 = env.reset()
    status = 1
    while(status==1):
        j += 1
#         print(j)
        act_, crit_ = model(state1.detach()) # act_ and crit_ are tensors
        # step the environment
        act = act_.view(-1).tolist()
        state2, reward, done = env.step(act)
        exp = (state1, reward, state2, done)
        replay.append(exp)
        state1 = state2
        
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, r, s2, d) in minibatch])
            reward_batch = reward_batch.to(device, dtype=torch.float)
            state2_batch = torch.cat([s2 for (s1, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, r, s2, d) in minibatch])
            done_batch = done_batch.to(device, dtype=torch.float)
            _, Q1 = model(state1_batch)
            with torch.no_grad():
                _, Q2 = MainModel(state2_batch.detach())
            Y = reward_batch + gamma * ((1 - done_batch) * Q2)
            X = Q1
            # print(X, Y.detach()) # 256,1 | 256, 256
            loss = loss_fn(X, Q2)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            losses.append(np.sqrt(loss.item()))
            optimizer.step()
            if j % syn_freq == 0:
                print(j)
                MainModel.load_state_dict(model.state_dict())
                torch.save(actor_model, './Checkpoints/ActorModel.pth')
                torch.save(critic_model, './Checkpoints/CriticModel.pth')
                torch.save(model, './Checkpoints/Model1.pth')
                torch.save(MainModel, './Checkpoints/MainModel.pth')
                if j == 5000:
                  done = True
        if done:
            status = 0

losses = np.array(losses)
plt.plot(losses)
plt.savefig("./Graphs/RL_Loss.png")