#Part 1 - Data Preprocessing

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
X_df = pd.DataFrame(X)
Y = dataset.iloc[:, -1].values
Y_df = pd.DataFrame(Y)

#Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
LabelEncoder_X_1 = LabelEncoder()
X[:,1] = LabelEncoder_X_1.fit_transform(X[:,1])
LabelEncoder_X_2 = LabelEncoder()
X[:,2] = LabelEncoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Part 2 - Making the ANN

#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer = 'uniform'))

#Adding second hidden layer
classifier.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))

#Adding the output layer
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 20)

#Part 3 - Making the predictions and evaluating the model

#Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#Making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
