import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
from collections import Counter

from DatasetsPreprocess.KNN_Preprocess import knn_dataset

class KNeighborsClasssifier():
    '''
        K-Nearest Neighbour Classifier
    '''
    def __init__(self, data, predict, k=1):
        self.data = data
        self.k = k
        self.predict = predict

    def euclidean(features, predict):
        # Euclidean distance between points a & data
        return np.linalg.norm(np.array(features) - np.array(predict))

    def fit(self):
        distance = []
        for group in self.data:
            for features in self.data[group]:
                euclidean_distance = self.euclidean(features, self.predict)
                distance.append([euclidean_distance, group])
        self.votes = [i[1] for i in sorted(distance) [:k]]
        
    def predict(self):
        vote_result = Counter(self.votes).most_common(1)[0][0]
        confidence = (Counter(self.votes).most_common(1)[0][1])/k
        return vote_result, confidence

# Getting the Training and Testing Data and splitting it
X, y, validation_set, data = knn_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Classing the Model to find the accuracy
try:
    correct = 0
    total = 0
    for group in y:
        for data in y[group]:
            knnc = KNeighborsClasssifier(X, data, k=5)
            vote, confidence = knnc.fit()
            if group == vote:
                correct += 1
    total +=1
    print('Accuracy: ', correct/total)
except:
    pass

# Testing the model and saving the data
k = 1
knn = KNeighborsClassifier(n_neighbors = k)
knn = knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print("Accuarcy of the KNN Model: ", accuracy)

# Predicting the values
result_freq = knn.predict(validation_set)
data['article'] = result_freq.tolist()

# Saving the Results into a CSV File
result_dataset = data.groupby('customer_id').head(12)
result_dataset1 = result_dataset.drop_duplicates(subset = ['customer_id','article'])
result_dataset2 = result_dataset1[['customer_id','article']]
result_dataset2['article'] = ' 0' + result_dataset1['article'].astype('str') 
result = result_dataset2.groupby('customer_id').sum().reset_index()
result.columns = ['customer_id','predictions']
result.to_csv("KNN_Results.csv", index=False, columns=["customer_id", "prediction"])
print(result.head())