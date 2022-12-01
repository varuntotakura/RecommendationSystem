import numpy as np
import pandas as pd

def knn_dataset():
    customers = pd.read_csv('h-and-m-personalized-fashion-recommendations/customers.csv')
    customers = customers[['customer_id','age','fashion_news_frequency','club_member_status']]

    articles = pd.read_csv('h-and-m-personalized-fashion-recommendations/articles.csv')
    articles = articles[['article_id','product_code','product_type_no','colour_group_code','section_no','garment_group_no']]
    articles['article_id'] = articles.article_id.astype('int32')
    
    transactions = pd.read_csv('h-and-m-personalized-fashion-recommendations/transactions_train.csv')
    transactions['year'] = pd.DatetimeIndex(transactions['t_dat']).year
    transactions['month'] = pd.DatetimeIndex(transactions['t_dat']).month
    transactions_per_year = transactions[['year','customer_id']].value_counts().reset_index()
    transactions_per_year.columns=['year', 'customer_id', 'num_of_purchase']

    frequent_trans = transactions_per_year[(transactions_per_year['num_of_purchase'] > 29)&(transactions_per_year['year'] == 2020)]
    frequent_trans_info = transactions[transactions['customer_id'].isin(frequent_trans['customer_id'])]
    ran_frequent_trans_info = frequent_trans_info.sample(n=50000)
    frequent_customers=customers[customers['customer_id'].isin(frequent_trans['customer_id'])]
    frequent_customers.loc[frequent_customers['fashion_news_frequency'] == 'NONE','fashion_news_frequency'] = float(0.0)
    frequent_customers.loc[frequent_customers['fashion_news_frequency'] == 'Regularly','fashion_news_frequency'] = float(0.5)
    frequent_customers.loc[frequent_customers['fashion_news_frequency'] == 'Monthly','fashion_news_frequency'] = float(1.0)
    frequent_customers.loc[frequent_customers['club_member_status'] == 'PRE-CREATE','club_member_status'] = float(0.5)
    frequent_customers.loc[frequent_customers['club_member_status'] == 'LEFT CLUB','club_member_status'] = float(0.0)
    frequent_customers.loc[frequent_customers['club_member_status'] == 'ACTIVE','club_member_status'] = float(1.0)
    frequent_info = ran_frequent_trans_info.merge(frequent_customers,on = 'customer_id').merge(articles,on = 'article_id')
    frequent_info_r = frequent_trans_info.merge(frequent_customers,on = 'customer_id').merge(articles,on = 'article_id')
    frequent_info = frequent_info.dropna()
    frequent_info.loc[frequent_info['fashion_news_frequency'] == 'NONE','fashion_news_frequency'] = float(0.0)
    frequent_info.loc[frequent_info['fashion_news_frequency'] == 'Regularly','fashion_news_frequency'] = float(0.5)
    frequent_info.loc[frequent_info['fashion_news_frequency'] == 'Monthly','fashion_news_frequency'] = float(1.0)
    frequent_info.loc[frequent_info['club_member_status'] == 'PRE-CREATE','club_member_status'] = float(0.5)
    frequent_info.loc[frequent_info['club_member_status'] == 'LEFT CLUB','club_member_status'] = float(0.0)
    frequent_info.loc[frequent_info['club_member_status'] == 'ACTIVE','club_member_status'] = float(1.0)
    frequent_info['fashion_news_frequency'] = frequent_info['fashion_news_frequency'].astype('float64')
    frequent_info['club_member_status'] = frequent_info['club_member_status'].astype('float64')
    frequent_info_r['fashion_news_frequency'] = frequent_info_r['fashion_news_frequency'].astype('float64')
    frequent_info_r['club_member_status'] = frequent_info_r['club_member_status'].astype('float64')
    frequent_info['colour_group_code_scale'] = (frequent_info['colour_group_code']-frequent_info['colour_group_code'].min())/(frequent_info['colour_group_code'].max()-frequent_info['colour_group_code'].min())
    frequent_info['garment_group_no_scale'] = (frequent_info['garment_group_no']-frequent_info['garment_group_no'].min())/(frequent_info['garment_group_no'].max()-frequent_info['garment_group_no'].min())
    frequent_info['product_type_no_scale'] = (frequent_info['product_type_no']-frequent_info['product_type_no'].min())/(frequent_info['product_type_no'].max()-frequent_info['product_type_no'].min())
    frequent_info['month_scale'] = (frequent_info['month']-frequent_info['month'].min())/(frequent_info['month'].max()-frequent_info['month'].min())
    frequent_info['age_scale'] = (frequent_info['age']-frequent_info['age'].min())/(frequent_info['age'].max()-frequent_info['age'].min())
    frequent_info_r['colour_group_code_scale'] = (frequent_info_r['colour_group_code']-frequent_info_r['colour_group_code'].min())/(frequent_info_r['colour_group_code'].max()-frequent_info_r['colour_group_code'].min())
    frequent_info_r['garment_group_no_scale'] = (frequent_info_r['garment_group_no']-frequent_info_r['garment_group_no'].min())/(frequent_info_r['garment_group_no'].max()-frequent_info_r['garment_group_no'].min())
    frequent_info_r['product_type_no_scale'] = (frequent_info_r['product_type_no']-frequent_info_r['product_type_no'].min())/(frequent_info_r['product_type_no'].max()-frequent_info_r['product_type_no'].min())
    frequent_info_r['month_scale'] = (frequent_info_r['month']-frequent_info_r['month'].min())/(frequent_info_r['month'].max()-frequent_info_r['month'].min())
    frequent_info_r['age_scale'] = (frequent_info_r['age']-frequent_info_r['age'].min())/(frequent_info_r['age'].max()-frequent_info_r['age'].min())
    frequent_info_r_model = frequent_info_r[['age_scale','colour_group_code_scale','garment_group_no_scale','fashion_news_frequency','product_type_no_scale','month_scale','club_member_status']]
    
    X = frequent_info[['age_scale','colour_group_code_scale','garment_group_no_scale','fashion_news_frequency','product_type_no_scale','month_scale','club_member_status']]
    y = frequent_info['article_id']
    frequent_info_r_model = frequent_info_r_model.dropna()
    results_prep_info_r = frequent_info_r.dropna()
    
    return X, y, frequent_info_r_model, results_prep_info_r