# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dataset = pd.read_csv('frb_01_overall_new2.csv')
X = dataset.iloc[:, 4:].values
y = dataset.iloc[:, 3].values

#Encode categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1 = LabelEncoder()
labelencoder_x2 = LabelEncoder()
labelencoder_x3 = LabelEncoder()
labelencoder_x4 = LabelEncoder()
X[:,0] = labelencoder_x1.fit_transform(X[:,0])
X[:,1] = labelencoder_x2.fit_transform(X[:,1])
X[:,2] = labelencoder_x3.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [2,3])
X = onehotencoder.fit_transform(X).toarray()
# Encode target classes to discrete values
labelEncoder = preprocessing.LabelEncoder()

# Dictionary for encoding
labelEncoder.fit(y)
riskyness = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
#print('Dictionary: '+ str(riskyness) + '\n')

y = labelEncoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Train classifier with gini method
classifier_gini = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=350)

#Fit training data to classifier
classifier_gini = classifier_gini.fit(X_train,y_train)

#Predict with gini method
y_predict_gini = classifier_gini.predict(X_test)

#Train classifier with entropy method
classifier_entropy = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=350)
classifier_entropy = classifier_entropy.fit(X_train,y_train)

#Predict with entropy method
y_predict_entropy = classifier_entropy.predict(X_test)

accuracy_gini = accuracy_score(y_test,y_predict_gini)
accuracy_entropy = accuracy_score(y_test,y_predict_entropy)

print('Accuracy score gini method: '+str(accuracy_gini))
print('Accuracy score entropy method: '+str(accuracy_entropy))


dot_data = StringIO()
export_graphviz(classifier_gini, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())



























