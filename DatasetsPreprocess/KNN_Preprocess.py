import numpy as np
import pandas as pd

def knn_dataset():
    '''
        Preprocess the dataset to feed in to the KNN Model so that only the features which are required can be used during the training of the model
    '''
    # Import Customer Dataset
    customers_data = pd.read_csv('h-and-m-personalized-fashion-recommendations/customers_data.csv')
    customers_data = customers_data[['customer_id','age','fashion_news_frequency','club_member_status']]

    # Import articles_data Dataset
    articles_data = pd.read_csv('h-and-m-personalized-fashion-recommendations/articles_data.csv')
    articles_data = articles_data[['article_id','product_code','product_type_no','colour_group_code','section_no','garment_group_no']]
    articles_data['article_id'] = articles_data.article_id.astype('int32')
    
    # Import transactions Dataset
    transactions = pd.read_csv('h-and-m-personalized-fashion-recommendations/transactions_train.csv')
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
    frequent_customers_data = customers_data[customers_data['customer_id'].isin(frequent_trans['customer_id'])]
    frequent_customers_data.loc[frequent_customers_data['fashion_news_frequency'] == 'NONE','fashion_news_frequency'] = float(0.0)
    frequent_customers_data.loc[frequent_customers_data['fashion_news_frequency'] == 'Regularly','fashion_news_frequency'] = float(0.5)
    frequent_customers_data.loc[frequent_customers_data['fashion_news_frequency'] == 'Monthly','fashion_news_frequency'] = float(1.0)
    frequent_customers_data.loc[frequent_customers_data['club_member_status'] == 'PRE-CREATE','club_member_status'] = float(0.5)
    frequent_customers_data.loc[frequent_customers_data['club_member_status'] == 'LEFT CLUB','club_member_status'] = float(0.0)
    frequent_customers_data.loc[frequent_customers_data['club_member_status'] == 'ACTIVE','club_member_status'] = float(1.0)
    frequent_data = random_frequent_transactions_info.merge(frequent_customers_data,on = 'customer_id').merge(articles_data,on = 'article_id')
    sampled_frequent_data = frequent_trans_info.merge(frequent_customers_data,on = 'customer_id').merge(articles_data,on = 'article_id')
    frequent_data = frequent_data.dropna()
    frequent_data.loc[frequent_data['fashion_news_frequency'] == 'NONE','fashion_news_frequency'] = float(0.0)
    frequent_data.loc[frequent_data['fashion_news_frequency'] == 'Regularly','fashion_news_frequency'] = float(0.5)
    frequent_data.loc[frequent_data['fashion_news_frequency'] == 'Monthly','fashion_news_frequency'] = float(1.0)
    frequent_data.loc[frequent_data['club_member_status'] == 'PRE-CREATE','club_member_status'] = float(0.5)
    frequent_data.loc[frequent_data['club_member_status'] == 'LEFT CLUB','club_member_status'] = float(0.0)
    frequent_data.loc[frequent_data['club_member_status'] == 'ACTIVE','club_member_status'] = float(1.0)
    frequent_data['fashion_news_frequency'] = frequent_data['fashion_news_frequency'].astype('float64')
    frequent_data['club_member_status'] = frequent_data['club_member_status'].astype('float64')
    sampled_frequent_data['fashion_news_frequency'] = sampled_frequent_data['fashion_news_frequency'].astype('float64')
    sampled_frequent_data['club_member_status'] = sampled_frequent_data['club_member_status'].astype('float64')
    frequent_data['colour_group_code_scale'] = (frequent_data['colour_group_code']-frequent_data['colour_group_code'].min())/(frequent_data['colour_group_code'].max()-frequent_data['colour_group_code'].min())
    frequent_data['garment_group_no_scale'] = (frequent_data['garment_group_no']-frequent_data['garment_group_no'].min())/(frequent_data['garment_group_no'].max()-frequent_data['garment_group_no'].min())
    frequent_data['product_type_no_scale'] = (frequent_data['product_type_no']-frequent_data['product_type_no'].min())/(frequent_data['product_type_no'].max()-frequent_data['product_type_no'].min())
    frequent_data['month_scale'] = (frequent_data['month']-frequent_data['month'].min())/(frequent_data['month'].max()-frequent_data['month'].min())
    frequent_data['age_scale'] = (frequent_data['age']-frequent_data['age'].min())/(frequent_data['age'].max()-frequent_data['age'].min())
    sampled_frequent_data['colour_group_code_scale'] = (sampled_frequent_data['colour_group_code']-sampled_frequent_data['colour_group_code'].min())/(sampled_frequent_data['colour_group_code'].max()-sampled_frequent_data['colour_group_code'].min())
    sampled_frequent_data['garment_group_no_scale'] = (sampled_frequent_data['garment_group_no']-sampled_frequent_data['garment_group_no'].min())/(sampled_frequent_data['garment_group_no'].max()-sampled_frequent_data['garment_group_no'].min())
    sampled_frequent_data['product_type_no_scale'] = (sampled_frequent_data['product_type_no']-sampled_frequent_data['product_type_no'].min())/(sampled_frequent_data['product_type_no'].max()-sampled_frequent_data['product_type_no'].min())
    sampled_frequent_data['month_scale'] = (sampled_frequent_data['month']-sampled_frequent_data['month'].min())/(sampled_frequent_data['month'].max()-sampled_frequent_data['month'].min())
    sampled_frequent_data['age_scale'] = (sampled_frequent_data['age']-sampled_frequent_data['age'].min())/(sampled_frequent_data['age'].max()-sampled_frequent_data['age'].min())
    sampled_frequent_data_model = sampled_frequent_data[['age_scale','colour_group_code_scale','garment_group_no_scale','fashion_news_frequency','product_type_no_scale','month_scale','club_member_status']]
    
    # Declare X and y Variables for the Training and Testing
    X = frequent_data[['age_scale','colour_group_code_scale','garment_group_no_scale','fashion_news_frequency','product_type_no_scale','month_scale','club_member_status']]
    y = frequent_data['article_id']
    sampled_frequent_data_model = sampled_frequent_data_model.dropna()
    results_prep_info_r = sampled_frequent_data.dropna()
    
    return X, y, sampled_frequent_data_model, results_prep_info_r
