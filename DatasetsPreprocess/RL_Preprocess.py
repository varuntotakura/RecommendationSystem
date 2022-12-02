<<<<<<< HEAD
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
=======
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
>>>>>>> cf1a8fdd59182a66ffe5b6732f8f0559e449c2ce
    return df_transactions, df_customers, df_products