import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Reading the data from csv file
heart_data = pd.read_csv('/AIML_PROJECT/HeartDiseasePrediction/heart_disease_data.csv')

# <---------------------------------->
# First five row of the data set
# print(heart_data.head())


# <---------------------------------->
# Number of rows and column
# print(heart_data.shape)


# <---------------------------------->
# Statistical measures about the data
# print(heart_data.describe())


# <---------------------------------->
# This will tell us how many 1&0
# print(heart_data['target'].value_counts())


# <---------------------------------->
X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']
# print(X)
# print(Y)


# <---------------------------------->
# Spliting the data into Training data & Test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# print(X.shape, X_train.shape, X_test.shape)


# <---------------------------------->
# Model Training 
model = LogisticRegression(max_iter=1000) 
model.fit(X_train,Y_train)


# # <---------------------------------->
# # Accuracy Score on training data
# X_train_prediction = model.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
# print('Accuracy on training data: ',training_data_accuracy)


# # <---------------------------------->
# # Accuracy Score on test data
# X_test_prediction = model.predict(X_test)
# test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
# print('Accuracy on test data: ',test_data_accuracy)


# <---------------------------------->
# Building a predective system
input_data = (56,1,1,120,236,0,1,178,0,0.8,2,0,2)
input_data_as_numpy_array = np.asarray(input_data)
# reshape the data
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
predection = model.predict(input_data_reshaped)

if(predection[0]==1):
    print("This person is suffering from Heart Disease.")
else:
    print("This person is not suffering from Heart Disease.")
