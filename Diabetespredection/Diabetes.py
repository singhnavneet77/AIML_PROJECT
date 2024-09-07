import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



# Loading the diabetes data to panda dataset
diabetes_dataset = pd.read_csv('/AIML_PROJECT/Diabetespredection/Diabetes.csv')

# <-------------------------------->
# printing the first five dataset
# print(diabetes_dataset.head())

# <-------------------------------->
# Number of rows and cloumns
# print(diabetes_dataset.shape)

# <-------------------------------->
# Getting the statistical measures of the data
# print(diabetes_dataset.describe())
# print(diabetes_dataset['Outcome'].value_counts())
# print(diabetes_dataset.groupby('Outcome').mean())  # This will give the mean value of both case

# <-------------------------------->
# Saperating the data and level
X = diabetes_dataset.drop(columns = 'Outcome',axis=1)
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)

# <-------------------------------->
# Standardizethe data
scaler = StandardScaler()
scaler.fit(X)
standardize_data = scaler.transform(X)
# print(standardize_data)

X = standardize_data
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)

# <-------------------------------->
# Train Test spilt
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)
# print(X.shape,X_train.shape,X_test.shape)

# <-------------------------------->
# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

# <-------------------------------->
# Accuracy score on train data
X_train_predection = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predection,Y_train)
# print("Accuracy score of the training model : ",training_data_accuracy)


# <-------------------------------->
# Accuracy score on test data
X_test_predection = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predection,Y_test)
# print("Accuracy score of the test model : ",test_data_accuracy)

# <-------------------------------->
# Making a predictive system
input_data = (0,137,40,35,168,43.1,2.288,33)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# standardize the input data
std_data = scaler.transform(input_data_reshaped)
# print(std_data)

predection = classifier.predict(std_data)
# print(predection)

if(predection[0]==0):
    print("The person is not diabetes")
else:
    print("The person is suffring from diabetes")
    



