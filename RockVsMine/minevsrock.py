# Importing Dependency

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Data Processing

sonar_data=pd.read_csv('E:/AIML_PROJECT/RockVsMine/Sonarminedata.csv',header=None)

data = sonar_data.head()
# print(data)
# print("\n")

# <------------------------------------------>
# Number of Rows and Columns
# print(sonar_data.shape)
# print("\n")

# <------------------------------------------>
# Describe statistical measures of data
# print(sonar_data.describe())
# print("\n")

# <------------------------------------------>
# Finding how many example of rock and mine
# print(sonar_data[60].value_counts())
# print("\n")

# <------------------------------------------>
# Mean for each of the data
# print(sonar_data.groupby(60).mean())
# print("\n")

# <------------------------------------------>
# Seprating data and Lable
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
# print(X)
# print(Y)

# <------------------------------------------>
# Training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1) 
# print(X.shape,X_train.shape,X_test.shape)

# <------------------------------------------>
# Below are training data
# print(X_train)
# print(Y_train)

# <------------------------------------------>
# Model training using Logistic regression
model = LogisticRegression()
data = model.fit(X_train,Y_train)
# print(data)

# <------------------------------------------>
# Model evalution ->accuracy on training data
X_train_predection = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predection,Y_train)
# print("Accuracy on training data : ",training_data_accuracy)

# <------------------------------------------>
# Accuracy on test data
X_test_predection = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predection,Y_test)
# print("Accuracy on test data : ",test_data_accuracy)

# <------------------------------------------>
# Making a predective data
input_data = (0.1021,0.0830,0.0577,0.0627,0.0635,0.1328,0.0988,0.1787,0.1199,0.1369,0.2509,0.2631,0.2796,0.2977,0.3823,0.3129,0.3956,0.2093,0.3218,0.3345,0.3184,0.2887,0.3610,0.2566,0.4106,0.4591,0.4722,0.7278,0.7591,0.6579,0.7514,0.6666,0.4903,0.5962,0.6552,0.4014,0.1188,0.3245,0.3107,0.1354,0.5109,0.7988,0.7517,0.5508,0.5858,0.7292,0.5522,0.3339,0.1608,0.0475,0.1004,0.0709,0.0317,0.0309,0.0252,0.0087,0.0177,0.0214,0.0227,0.0106)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
predection = model.predict(input_data_reshaped)
# print(predection)
if(predection[0]=='R'):
    print("The object is a Rock")
else:
    print("The object is Mine")