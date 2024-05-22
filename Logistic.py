# importing required libraries
# analyze UCI ML Breast Cancer Wisconsin (Diagnostic) dataset
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the breast cancer dataset
X,y = load_breast_cancer(return_X_y = True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fit training data to logistic regression model
log_regression = LogisticRegression(penalty=None).fit(X_train, y_train)
y_pred = log_regression.predict(X_test)

# print test accuracy
print("%.3f" % accuracy_score(y_test, y_pred))

# Adjust regularization coefficient (L2 / lasso regularizer)
#https://www.google.com/url?q=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fgenerated%2Fsklearn.linear_model.LogisticRegression.html

accuracy_list = []
C_list = np.array([0.001, 0.01, 0.1, 1, 10, 100])
lambda_list = 1 / C_list


# iterate over vals of C, testing logistic regression with diff values
for c in C_list:
    log_regr = LogisticRegression(C = c).fit(X_train, y_train)
    y_pred = log_regr.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    print("C = %f" % c)
    print("Test accuracy = %.3f" %  accuracy_score(y_test, y_pred))


#plot
fig, ax = plt.subplots()
ax.plot(lambda_list, accuracy_list)

ax.set_xlabel('lambda')
ax.set_ylabel("test accuracy")
