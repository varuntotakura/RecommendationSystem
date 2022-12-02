import pandas as pd
def RL_Dataset():
    '''
        To get the data files and preprocess it before using
    '''
    # Transactions Data
    transactions_train = '../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv'
    df_transactions = pd.read_csv(transactions_train)
    df_transactions['customer_id'] = df_transactions.customer_id.astype('str')
    df_transactions['article_id'] = df_transactions.article_id.astype('str')
    df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
    df_transactions = df_transactions.set_index('t_dat')

    # Customers Data
    customers = '../input/h-and-m-personalized-fashion-recommendations/customers.csv'
    df_customers = pd.read_csv(customers)
    df_customers['customer_id'] = df_customers.customer_id.astype('str')

    # Arcticles Data
    articles = '../input/h-and-m-personalized-fashion-recommendations/articles.csv'
    df_articles = pd.read_csv(articles)
    df_articles['article_id'] = df_articles.article_id.astype('str')

    return df_transactions, df_customers, df_articles