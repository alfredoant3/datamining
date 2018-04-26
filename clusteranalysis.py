# -*- coding: utf-8 -*-
#%matplotlib inline
import matplotlib.pyplot
from matplotlib import style
style.use('ggplot')
import pandas
import numpy 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from matplotlib.pyplot import subplot


dataset = pandas.read_csv('frb_01_overall_new2.csv')
X = dataset.iloc[:,17:18].values  # % Borrowers without a Pell Grant
y = dataset.iloc[:, 20].values  # Mean Balance

Xy = numpy.array(list(zip(X,y)))
kmeans = KMeans(n_clusters = 5)
kmeans.fit(Xy)
fig1 = matplotlib.pyplot.subplot()
centers = kmeans.cluster_centers_
labels = kmeans.labels_
colors = ['r.','g.','b.','c.','k.','y.','m.']
#matplotlib.pyplot.scatter(X,y)
print('Displaying: % Borrowers without a Pell Grant vs. Mean Balance')
fig1.scatter(centers[:, 0], centers[:, 1],zorder=1, c='black', s=100,alpha=0.85)
for i in range(len(Xy)):
    fig1.plot(Xy[i][0],Xy[i][1],
                           colors[labels[i]],
                           markersize=10)
    
fig1.set_title('% Borrowers without a Pell Grant vs. Mean Balance')
fig1.set_xlabel('% Borrowers without a Pell Grant')
fig1.set_ylabel('Mean Balance')
fig1.show()

#print('Displaying: Median Dependent Parent AGI vs. Mean Balance')
#
#X = dataset.iloc[:,14].values  # Median Dependent Parent AGI
#
#
##fig2
#print('Displaying: % Borrowers without a Pell Grant vs. Mean Balance')
#fig1.scatter(centers[:, 0], centers[:, 1],zorder=1, c='black', s=100,alpha=0.85)
#for i in range(len(Xy)):
#    fig1.plot(Xy[i][0],Xy[i][1],
#                           colors[labels[i]],
#                           markersize=10)
#    
#fig1.set_title('% Borrowers without a Pell Grant vs. Mean Balance')
#fig1.set_xlabel('% Borrowers without a Pell Grant')
#fig1.set_ylabel('Mean Balance')
#fig1.show()














































