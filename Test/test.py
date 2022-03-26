import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Test/australian.dat', sep=' ')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

print(kmeans.predict([[0.0, 47.42, 8.0, 2.0, 10.0, 5.0, 6.5, 1.0, 1.0, 6.0, 0.0, 2.0, 375.0, 51101.0]]))
print(kmeans.predict([[0.0, 21.67, 11.5, 1.0, 5.0, 3.0, 0.0, 1.0, 1.0, 11.0, 1.0, 2.0, 0.0, 1.0]]))


