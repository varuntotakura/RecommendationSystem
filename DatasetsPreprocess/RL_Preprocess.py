import numpy as np
import pandas as pd

def RL_Dataset():
    transactions_train = '../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv'
    df_transactions = pd.read_csv(transactions_train,
                                dtype= {
                                    'customer_id': 'str',
                                    'article_id': 'str'
                                })
    df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
    df_transactions = df_transactions.set_index('t_dat')
    customers = '../input/h-and-m-personalized-fashion-recommendations/customers.csv'
    df_customers = pd.read_csv(customers,
                                dtype= {
                                    'customer_id': 'str'
                                })
    products = '../input/h-and-m-personalized-fashion-recommendations/articles.csv'
    df_products = pd.read_csv(products,
                                dtype= {
                                    'article_id': 'str'
                                })
    return df_transactions, df_customers, df_products