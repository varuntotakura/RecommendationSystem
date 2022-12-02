import numpy as np
import pandas as pd

def knn_dataset():
    '''
        Preprocess the dataset to feed in to the KNN Model so that only the features which are required can be used during the training of the model
    '''
    # Import Customer Dataset
    customers = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/customers.csv')
    customers = customers[['customer_id','age','fashion_news_frequency','club_member_status']]

    # Import Articles Dataset
    articles = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/articles.csv')
    articles = articles[['article_id','product_code','product_type_no','colour_group_code','section_no','garment_group_no']]
    articles['article_id'] = articles.article_id.astype('int32')
    
    # Import transactions Dataset
    transactions = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
    transactions['year'] = pd.DatetimeIndex(transactions['t_dat']).year
    transactions['month'] = pd.DatetimeIndex(transactions['t_dat']).month
    # Get transactions per year
    transactions_per_year = transactions[['year','customer_id']].value_counts().reset_index()
    transactions_per_year.columns=['year', 'customer_id', 'num_of_purchase']
    # Get the frequent transactions
    frequent_trans = transactions_per_year[(transactions_per_year['num_of_purchase'] > 29)&(transactions_per_year['year'] == 2020)]
    frequent_trans_info = transactions[transactions['customer_id'].isin(frequent_trans['customer_id'])]
    # Get a Sample of frequent transactions
    # Get each transaction and get the features is of the transactions
    random_frequent_transactions_info = frequent_trans_info.sample(n=50000)
    frequent_customers = customers[customers['customer_id'].isin(frequent_trans['customer_id'])]
    frequent_customers.loc[frequent_customers['fashion_news_frequency'] == 'NONE','fashion_news_frequency'] = float(0.0)
    frequent_customers.loc[frequent_customers['fashion_news_frequency'] == 'Regularly','fashion_news_frequency'] = float(0.5)
    frequent_customers.loc[frequent_customers['fashion_news_frequency'] == 'Monthly','fashion_news_frequency'] = float(1.0)
    frequent_customers.loc[frequent_customers['club_member_status'] == 'PRE-CREATE','club_member_status'] = float(0.5)
    frequent_customers.loc[frequent_customers['club_member_status'] == 'LEFT CLUB','club_member_status'] = float(0.0)
    frequent_customers.loc[frequent_customers['club_member_status'] == 'ACTIVE','club_member_status'] = float(1.0)
    frequent_info = random_frequent_transactions_info.merge(frequent_customers,on = 'customer_id').merge(articles,on = 'article_id')
    sampled_frequent_info = frequent_trans_info.merge(frequent_customers,on = 'customer_id').merge(articles,on = 'article_id')
    frequent_info = frequent_info.dropna()
    frequent_info.loc[frequent_info['fashion_news_frequency'] == 'NONE','fashion_news_frequency'] = float(0.0)
    frequent_info.loc[frequent_info['fashion_news_frequency'] == 'Regularly','fashion_news_frequency'] = float(0.5)
    frequent_info.loc[frequent_info['fashion_news_frequency'] == 'Monthly','fashion_news_frequency'] = float(1.0)
    frequent_info.loc[frequent_info['club_member_status'] == 'PRE-CREATE','club_member_status'] = float(0.5)
    frequent_info.loc[frequent_info['club_member_status'] == 'LEFT CLUB','club_member_status'] = float(0.0)
    frequent_info.loc[frequent_info['club_member_status'] == 'ACTIVE','club_member_status'] = float(1.0)
    frequent_info['fashion_news_frequency'] = frequent_info['fashion_news_frequency'].astype('float64')
    frequent_info['club_member_status'] = frequent_info['club_member_status'].astype('float64')
    sampled_frequent_info['fashion_news_frequency'] = sampled_frequent_info['fashion_news_frequency'].astype('float64')
    sampled_frequent_info['club_member_status'] = sampled_frequent_info['club_member_status'].astype('float64')
    frequent_info['colour_group_code_scale'] = (frequent_info['colour_group_code']-frequent_info['colour_group_code'].min())/(frequent_info['colour_group_code'].max()-frequent_info['colour_group_code'].min())
    frequent_info['garment_group_no_scale'] = (frequent_info['garment_group_no']-frequent_info['garment_group_no'].min())/(frequent_info['garment_group_no'].max()-frequent_info['garment_group_no'].min())
    frequent_info['product_type_no_scale'] = (frequent_info['product_type_no']-frequent_info['product_type_no'].min())/(frequent_info['product_type_no'].max()-frequent_info['product_type_no'].min())
    frequent_info['month_scale'] = (frequent_info['month']-frequent_info['month'].min())/(frequent_info['month'].max()-frequent_info['month'].min())
    frequent_info['age_scale'] = (frequent_info['age']-frequent_info['age'].min())/(frequent_info['age'].max()-frequent_info['age'].min())
    sampled_frequent_info['colour_group_code_scale'] = (sampled_frequent_info['colour_group_code']-sampled_frequent_info['colour_group_code'].min())/(sampled_frequent_info['colour_group_code'].max()-sampled_frequent_info['colour_group_code'].min())
    sampled_frequent_info['garment_group_no_scale'] = (sampled_frequent_info['garment_group_no']-sampled_frequent_info['garment_group_no'].min())/(sampled_frequent_info['garment_group_no'].max()-sampled_frequent_info['garment_group_no'].min())
    sampled_frequent_info['product_type_no_scale'] = (sampled_frequent_info['product_type_no']-sampled_frequent_info['product_type_no'].min())/(sampled_frequent_info['product_type_no'].max()-sampled_frequent_info['product_type_no'].min())
    sampled_frequent_info['month_scale'] = (sampled_frequent_info['month']-sampled_frequent_info['month'].min())/(sampled_frequent_info['month'].max()-sampled_frequent_info['month'].min())
    sampled_frequent_info['age_scale'] = (sampled_frequent_info['age']-sampled_frequent_info['age'].min())/(sampled_frequent_info['age'].max()-sampled_frequent_info['age'].min())
    sampled_frequent_info_model = sampled_frequent_info[['age_scale','colour_group_code_scale','garment_group_no_scale','fashion_news_frequency','product_type_no_scale','month_scale','club_member_status']]
    
    # Declare X and y Variables for the Training and Testing
    X = frequent_info[['age_scale','colour_group_code_scale','garment_group_no_scale','fashion_news_frequency','product_type_no_scale','month_scale','club_member_status']]
    y = frequent_info['article_id']
    sampled_frequent_info_model = sampled_frequent_info_model.dropna()
    results_prep_info_r = sampled_frequent_info.dropna()
    
    return X, y, sampled_frequent_info_model, results_prep_info_r