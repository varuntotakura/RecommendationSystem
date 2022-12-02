import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch

class LSTMDataset(Dataset):
    '''
        Making the dataset modifications in the structure and values of the data elements so that it can be used while training and testing
    '''
    def __init__(self, transactions, sequence_length, test_purpose=False):
        self.transactions = transactions.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.test_purpose = test_purpose
    
    def __len__(self):
        '''
            To get the length of the transactions
        '''
        return self.transactions.shape[0]
    
    # To get the item of the specific transaction
    def __getitem__(self, index, WEEK_HIST_MAX = 5):
        transaction = self.transactions.iloc[index]
        if self.test_purpose:
            target_transaction = torch.zeros(2).float()
        else:
            if not transaction.target:
                target_transaction = torch.tensor([0]).int()
            else:
                random_target = np.random.choice(transaction.target,1)
                # Squeezing the data so that it will transform in to a shape of row of the transacation
                target_transaction = torch.tensor(random_target).squeeze().int()

        article_hist = torch.zeros(self.sequence_length).long()
        week_hist = torch.ones(self.sequence_length).float()
        
        if isinstance(transaction.article_id, list):
            if len(transaction.article_id) >= self.sequence_length:
                article_hist = torch.LongTensor(transaction.article_id[-self.sequence_length:])
                week_hist = (torch.LongTensor(transaction.week_history[-self.sequence_length:]) - transaction.week)/WEEK_HIST_MAX/2
            else:
                article_hist[-len(transaction.article_id):] = torch.LongTensor(transaction.article_id)
                week_hist[-len(transaction.article_id):] = (torch.LongTensor(transaction.week_history) - transaction.week)/WEEK_HIST_MAX/2
                
        return article_hist, week_hist, target_transaction

def format_dataset(transactions, week, WEEK_HIST_MAX = 5):
    '''
        Format the dataset so that it can be feeded to the model while training or testing
    '''
    # Getting the history of transactions
    history_df = transactions[(transactions["week"] > week) & (transactions["week"] <= week + WEEK_HIST_MAX)]
    history_df = history_df.groupby("customer_id").agg({"article_id": list, "week": list}).reset_index()
    history_df.rename(columns={"week": 'week_history'}, inplace=True)
    
    # Choosing only the transactions which are of specific week as mentioned in the parameter
    target_transactions = transactions[transactions["week"] == week]
    target_transactions = target_transactions.groupby("customer_id").agg({"article_id": list}).reset_index()
    target_transactions.rename(columns={"article_id": "target"}, inplace=True)
    target_transactions["week"] = week
    
    return target_transactions.merge(history_df, on="customer_id", how="left")

def LSTM_Dataset():
    '''
        To get the train and test set from the datasets available
    '''
    WEEK_HIST_MAX = 5

    # Import customer table
    customers = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/customers.csv')
    customers = customers[['customer_id','age','fashion_news_frequency','club_member_status']]

    # Import Articles table
    articles = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/articles.csv')
    articles = articles[['article_id','product_code','product_type_no','colour_group_code','section_no','garment_group_no']]
    articles['article_id'] = articles.article_id.astype('int32')
    
    # Import transactions table
    transactions = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    # Sorting the frequently bought items from the transactions table
    frequently_bought = transactions.groupby("article_id")["t_dat"].max().reset_index()
    # Taking a part of the data because of the computational complexity and challenges
    frequently_bought = frequently_bought[frequently_bought["t_dat"] >= "2020-09-01"].reset_index()
    
    # Taking only the transactions which are frequently done
    transactions = transactions[transactions["article_id"].isin(frequently_bought["article_id"])].reset_index(drop=True)

    # To get the week number of the transactions
    transactions["week"] = (transactions["t_dat"].max() - transactions["t_dat"]).dt.days // 7

    # Adding <Start> in the beggining of list to make sure that it has enough length to process
    article_ids = np.concatenate([["<Start>"], np.unique(transactions["article_id"].values)])

    # As there are lot of articles, encode with a specific value so that that can be decoded again when required
    label_encoder = LabelEncoder()
    label_encoder.fit(article_ids)
    transactions["article_id"] = label_encoder.fit_transform(transactions["article_id"])

    # Making train and test data
    train_weeks = [0, 1, 2, 3]
    val_weeks = [4]
    training_transactions = pd.concat([format_dataset(transactions, w) for w in train_weeks]).reset_index(drop=True)
    testing_transactions = pd.concat([format_dataset(transactions, w) for w in val_weeks]).reset_index(drop=True)
    n_classes = transactions["article_id"].nunique()+1

    return transactions, label_encoder, training_transactions, testing_transactions, n_classes