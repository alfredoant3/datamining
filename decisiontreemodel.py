# -*- coding: utf-8 -*-

import pandas
import numpy 
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy

school_data = pandas.read_csv('frb_01_overall_new.csv')

# X features are attributes that revealed a positive correlation with risk
# Median Dependent Parent AGI R-Value = 0.39
# % Borrowers without a Pell Grant R-Value = 0.47
# % Dependent Borrower Count R-Value = 0.21
# Mean Balance R-Value = 0.22


X_features = school_data.iloc[:,[11,14,17,20]]
y_target = school_data.iloc[:,3]

# Encode target classes to discrete values
labelEncoder = preprocessing.LabelEncoder()

# Dictionary for encoding
labelEncoder.fit(y_target)
riskyness = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
print('Dictionary: '+ str(riskyness) + '\n')

y_target = labelEncoder.fit_transform(y_target)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, 
                                                    test_size = 0.15, random_state = 0)

#Train classifier with gini method
classifier_gini = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=220)

#Fit training data to classifier
classifier_gini = classifier_gini.fit(X_train,y_train)

#Predict with gini method
y_predict_gini = classifier_gini.predict(X_test)

#Train classifier with entropy method
classifier_entropy = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=220)
classifier_entropy = classifier_gini.fit(X_train,y_train)

#Predict with entropy method
y_predict_entropy = classifier_entropy.predict(X_test)

accuracy_gini = accuracy_score(y_test,y_predict_gini)
accuracy_entropy = accuracy_score(y_test,y_predict_entropy)

print('Accuracy score gini method: '+str(accuracy_gini))
print('Accuracy score entropy method: '+str(accuracy_entropy))






























