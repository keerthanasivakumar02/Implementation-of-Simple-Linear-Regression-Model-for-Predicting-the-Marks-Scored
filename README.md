# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries for data handling, visualization, and model building.
2. Load the dataset and inspect the first and last few records to understand the data structure.
3. Prepare the data by separating the independent variable (hours studied) and the dependent variable (marks scored).

4. Split the dataset into training and testing sets to evaluate the model's performance.
5. Initialize and train a linear regression model using the training data.
6. Predict the marks for the test set using the trained model.
7. Evaluate the model by comparing the predicted marks with the actual marks from the test set.
8. Visualize the results for both the training and test sets by plotting the actual data points and the regression line

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KEERTHANA S
RegisterNumber:  212223040092


import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
##OUTPUT:
![Screenshot 2024-09-02 213048](https://github.com/user-attachments/assets/a26b3e28-17c1-435d-9e38-2afb3fd332f1)
```
dataset.info()
```
##Output:

![image](https://github.com/user-attachments/assets/13fafa5b-d17b-4ec8-9234-e37ff13e269c)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```

##Output:

![image](https://github.com/user-attachments/assets/e796acb4-8d26-4e47-a638-c371409cf5c4)

```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
```
##Output:
![image](https://github.com/user-attachments/assets/1cd7a5b0-1a38-44c0-a2ec-f3ea354a414d)

```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

```
##Output:
![image](https://github.com/user-attachments/assets/6a01e1d7-e0ae-4f8f-ae26-349db63da226)

```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)

```
##Output:
![image](https://github.com/user-attachments/assets/113b67c6-2626-4b8b-8114-8c82e60cb7b5)

```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="green")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
##Output:

![image](https://github.com/user-attachments/assets/aa1fb087-961d-48d8-ad37-3322ceb99716)

```
plt.scatter(X_test, Y_test,color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
##Output:
![image](https://github.com/user-attachments/assets/832cdfa4-3900-4f7e-b36b-76d2c8c3ea8b)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
