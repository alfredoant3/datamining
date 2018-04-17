# -*- coding: utf-8 -*-
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas
import numpy 
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

n=7
cols = [2,n]
for i in range(7,73):
    cols.append(n+1)
    n+=1

school_data = pandas.read_csv('frb_01_overall_new2.csv')
X = school_data.iloc[:,cols].values
#X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
#plt.scatter(X[:, 0], X[:, 17], s=50)


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);




















































