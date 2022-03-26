import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Test/australian_2.dat', sep=' ')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

print(kmeans.predict([[0.0, 47.42, 8.0, 2.0, 10.0, 5.0, 6.5, 1.0, 1.0, 6.0, 0.0, 2.0, 375.0, 51101.0]]))
print(kmeans.predict([[0.0, 21.67, 11.5, 1.0, 5.0, 3.0, 0.0, 1.0, 1.0, 11.0, 1.0, 2.0, 0.0, 1.0]]))


# Write a function that take a matrix of features x and applies k means clustering to it.
# The function should return the predicted labels of x.
# this fuction should't use the sklearn library.
def k_means(x, k):
    centroids = np.zeros((k, x.shape[1]))
    for i in range(k):
        centroids[i, :] = x[np.random.randint(x.shape[0]), :]
    labels = np.zeros(x.shape[0])
    distance = np.zeros((x.shape[0], k))
    convergence = False
    while not convergence:
        for i in range(x.shape[0]):
            for j in range(k):
                distance[i, j] = np.linalg.norm(x[i, :] - centroids[j, :])
            labels[i] = np.argmin(distance[i, :])
        for i in range(k):
            centroids[i, :] = np.mean(x[labels == i, :], axis=0)
        if np.all(labels == labels[0]):
            convergence = True
    return labels


labels = k_means(X, 2)
print(labels)
