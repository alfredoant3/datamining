# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

# Importing the dataset
dataset = pd.read_csv('frb_01_overall_new2.csv')
X = dataset.iloc[:, 4:].values
y = dataset.iloc[:, 3].values

#Encode categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
labelencoder_x2 = LabelEncoder()
labelencoder_x3 = LabelEncoder()
labelencoder_x4 = LabelEncoder()
X[:,0] = labelencoder_x1.fit_transform(X[:,0])
X[:,1] = labelencoder_x2.fit_transform(X[:,1])
X[:,2] = labelencoder_x3.fit_transform(X[:,2])
y = labelencoder_x4.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features = [2,3])
onehotencoder2 = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
y = y.reshape(-1,1)
y = onehotencoder2.fit_transform(y).toarray()
#X = X[:,1:]
y = y[:,0:4]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initialize ANN
classifier = keras.models.Sequential()

#Add input and first hidden layer
classifier.add(keras.layers.Dense(activation="relu", input_dim=82, units=13, kernel_initializer="uniform"))

#Add second hidden layer
classifier.add(keras.layers.Dense(activation="relu", input_dim=82, units=13, kernel_initializer="uniform"))

#Add third hidden layer
classifier.add(keras.layers.Dense(activation="relu", input_dim=82, units=13, kernel_initializer="uniform"))

#Add output layer
classifier.add(keras.layers.Dense(units = 4,activation = 'softmax',kernel_initializer = 'uniform'))

# Compile the ANN
classifier.compile(optimizer='adam',loss='categorical_crossentropy', metrics = ['accuracy'])

# Fit ANN to training set
classifier.fit(X_train, y_train,batch_size=10,epochs=275)

y_pred = classifier.predict(X_test)

# Convert probabilities to binaries.
for i in range(len(y_pred)):
    y_pred[i] = (y_pred[i] > 0.5)
    
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,y_pred)
print('Accuracy: ' +str(acc))


























