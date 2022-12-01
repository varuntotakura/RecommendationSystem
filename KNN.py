import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from DatasetsPreprocess.Preprocess import knn_dataset
from sklearn.metrics import accuracy_score
import pandas as pd

def most_common(lst):
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy

# # Unpack the iris dataset, from UCI Machine Learning Repository
# iris = datasets.load_iris()
# X = iris['data']
# y = iris['target']

# # Split data into train & test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# # Preprocess data
# ss = StandardScaler().fit(X_train)
# X_train, X_test = ss.transform(X_train), ss.transform(X_test)
# # Test knn model across varying ks
# accuracies = []
# ks = range(1, 30)
# for k in ks:
#     knn = KNeighborsClassifier(k=k)
#     knn.fit(X_train, y_train)
#     accuracy = knn.evaluate(X_test, y_test)
#     accuracies.append(accuracy)
# # Visualize accuracy vs. k
# fig, ax = plt.subplots()
# ax.plot(ks, accuracies)
# ax.set(xlabel="k",
#        ylabel="Accuracy",
#        title="Performance of knn")
# plt.show()


X, y, validation_set, data = knn_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

k = 1
knn = KNeighborsClassifier(n_neighbors = k)
knn = knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = accuracy_score(predictions, y_test)
print("Accuarcy of the KNN Model: ", accuracy)

result_freq = knn.predict(validation_set)

data['article'] = result_freq.tolist()

results_prep = data.groupby('customer_id').head(12)
results_prep1 = results_prep.drop_duplicates(subset = ['customer_id','article'])
results_prep2 = results_prep1[['customer_id','article']]
results_prep2['article'] = ' 0' + results_prep1['article'].astype('str') 
result = results_prep2.groupby('customer_id').sum().reset_index()
result.columns = ['customer_id','predictions']

print(result)