import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
# print(breast_cancer_dataset)

data_frame = pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)
# print(data_frame.head())

data_frame['label'] = breast_cancer_dataset.target
# print(data_frame.tail())

# Getting some information about data
# print(data_frame.info())

# Statistical measures about data
# print(data_frame.describe())

# Checking the distribution of target variable
# print(data_frame['label'].value_counts())
# print(data_frame.groupby('label').mean())

# Separating the features and target
X = data_frame.drop(columns='label',axis=1)
Y = data_frame['label']
# print(X)
# print(Y)

# Spliting the data into trainig data and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
# print(X.shape,X_train.shape,X_test.shape)


# Model Trainig -> Logistic regression
model = LogisticRegression()
model.fit(X_train,Y_train)

# Accuracy score on training data
X_train_predection = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_predection)
# print('Accuracy on training data: ',training_data_accuracy)

# Accuracy score on testing data
X_test_predection = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test,X_test_predection)
# print('Accuracy on testing data: ',testing_data_accuracy)


input_data = (15.71,13.93,102,761.7,0.09462,0.09462,0.07135,0.05933,0.1816,0.05723,0.3117,0.8155,1.972,27.94,0.005217,0.01515,0.01678,0.01268,0.01669,0.00233,17.5,19.25,114.3,922.8,0.1223,0.1949,0.1709,0.1374,0.2723,0.07071)
 
input_data_as_numpay_array = np.asanyarray(input_data)
input_data_reshaped = input_data_as_numpay_array.reshape(1,-1)
predection = model.predict(input_data_reshaped)
# 0->Malingant
# 1->Benign
print(predection)
if(predection[0]==0):
    print('The Breast Cancer is Malingant')
else:
    print('The Breast Cancer is Benign')