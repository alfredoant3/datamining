# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas
from matplotlib.pyplot import subplot
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# Importing the dataset
dataset = pandas.read_csv('frb_01_overall_new2.csv')
X = dataset.iloc[:, 4:].values
y = dataset.iloc[:, 3].values

# -------------------Encode categorical variables---------------------
from sklearn.preprocessing import LabelEncoder
labelencoder_x1 = LabelEncoder()
labelencoder_x2 = LabelEncoder()
labelencoder_x3 = LabelEncoder()
labelencoder_x4 = LabelEncoder()
X[:,0] = labelencoder_x1.fit_transform(X[:,0])
X[:,1] = labelencoder_x2.fit_transform(X[:,1])
X[:,2] = labelencoder_x3.fit_transform(X[:,2])

#--------------------Encode predtermined labels------------------------
y = labelencoder_x4.fit_transform(y)

#---------------TSNE dimensionality reduction--------------------------
X_embedded = TSNE(n_components=2).fit_transform(X)

#----------------Fit for K means clustering----------------------------
kmeans = KMeans(n_clusters = 5)
kmeans.fit(X_embedded)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

#-------------------------Plotting-------------------------------------
colors = ['r.','g.','b.','c.','y.','m.']

#plt.scatter(centers[:, 0], centers[:, 1],zorder=1, c='black', s=100,alpha=0.85)
for i in range(len(X_embedded)):
    plt.plot(X_embedded[i][0],X_embedded[i][1], colors[labels[i]], markersize=10)

plt.title('TSNE Dimension Reduction X vs. Y')
plt.xlabel('TSNE X Data')
plt.ylabel('TSNE Y Data')
plt.show()

#-----------------------Accuracy---------------------------------------
from sklearn.metrics import accuracy_score
acc = accuracy_score(y,labels)
print('Accuracy: ' +str(acc))




