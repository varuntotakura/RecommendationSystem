import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch

class LSTMDataset(Dataset):
    def __init__(self, transactions, sequence_length, is_test=False):
        self.transactions = transactions.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.is_test = is_test
    
    def __len__(self):
        return self.transactions.shape[0]
    
    def __getitem__(self, index, WEEK_HIST_MAX = 5):
        row = self.transactions.iloc[index]
        
        if self.is_test:
            target = torch.zeros(2).float()
        else:
            if not row.target:
                target = torch.tensor([0]).int()
            else:
                rand_target = np.random.choice(row.target,1)
                target = torch.tensor(rand_target).squeeze().int()

            
        article_hist = torch.zeros(self.sequence_length).long()
        week_hist = torch.ones(self.sequence_length).float()
        
        
        if isinstance(row.article_id, list):
            if len(row.article_id) >= self.sequence_length:
                article_hist = torch.LongTensor(row.article_id[-self.sequence_length:])
                week_hist = (torch.LongTensor(row.week_history[-self.sequence_length:]) - row.week)/WEEK_HIST_MAX/2
            else:
                article_hist[-len(row.article_id):] = torch.LongTensor(row.article_id)
                week_hist[-len(row.article_id):] = (torch.LongTensor(row.week_history) - row.week)/WEEK_HIST_MAX/2
                
        return article_hist, week_hist, target

def create_dataset(transactions, week, WEEK_HIST_MAX = 5):
    hist_transactions = transactions[(transactions["week"] > week) & (transactions["week"] <= week + WEEK_HIST_MAX)]
    hist_transactions = hist_transactions.groupby("customer_id").agg({"article_id": list, "week": list}).reset_index()
    hist_transactions.rename(columns={"week": 'week_history'}, inplace=True)
    
    target_transactions = transactions[transactions["week"] == week]
    target_transactions = target_transactions.groupby("customer_id").agg({"article_id": list}).reset_index()
    target_transactions.rename(columns={"article_id": "target"}, inplace=True)
    target_transactions["week"] = week
    
    return target_transactions.merge(hist_transactions, on="customer_id", how="left")

def LSTM_Dataset():
    WEEK_HIST_MAX = 5

    customers = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/customers.csv')
    customers = customers[['customer_id','age','fashion_news_frequency','club_member_status']]

    articles = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/articles.csv')
    articles = articles[['article_id','product_code','product_type_no','colour_group_code','section_no','garment_group_no']]
    articles['article_id'] = articles.article_id.astype('int32')
    
    transactions = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    active_articles = transactions.groupby("article_id")["t_dat"].max().reset_index()
    active_articles = active_articles[active_articles["t_dat"] >= "2019-09-01"].reset_index()
    
    transactions = transactions[transactions["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)
    transactions["week"] = (transactions["t_dat"].max() - transactions["t_dat"]).dt.days // 7

    article_ids = np.concatenate([["placeholder"], np.unique(transactions["article_id"].values)])

    le_article = LabelEncoder()
    le_article.fit(article_ids)
    transactions["article_id"] = le_article.fit_transform(transactions["article_id"])

    train_weeks = [0, 1, 2, 3]
    val_weeks = [4]
    train_transactions = pd.concat([create_dataset(transactions, w) for w in train_weeks]).reset_index(drop=True)
    val_transactions = pd.concat([create_dataset(transactions, w) for w in val_weeks]).reset_index(drop=True)
    n_classes = transactions["article_id"].nunique()+1
    return transactions, le_article, train_transactions, val_transactions, n_classes

