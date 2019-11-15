import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()

print('----prints everything----')
print(cancer)

print('----only the data ----')
print(cancer.data)

print('----shape of the data: 569, 30----')
print(cancer.data.shape)
cancer_data = np.c_[cancer.data, cancer.target]

print('----target names: malignant, benign----')
print(cancer.target_names)

print('----feature - we have 30 for col_name----')
col_name = cancer.feature_names
print(col_name)


col_name = cancer.feature_names
y = cancer.target
X = cancer.data

# Splitting features and target datasets into: train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=32)

# Training a Linear Regression model with fit() then lets see the Accuracy for Train and Test sets
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
print(f"log_reg - Accuracy for test: {log_reg.score(X_test, y_test):.4f}")
print(f"log_reg - Accuracy for train: {log_reg.score(X_train, y_train):.4f}")

# see the prediction for X_test[8]
predict = log_reg.predict([X_test[8]])
prediction = cancer.target_names[predict]
print(prediction)
print(predict)

# Printing the residuals: difference between real and predicted
# for (real, predicted) in list(zip(y_test, predictions)):
#     print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
# print(f'Accuracy score is {log_reg.score(X_test, y_test):.2f}/1 \n')

# Printing the classification report
from sklearn.metrics import classification_report
print('Classification Report')
print(classification_report(y_test, predictions))

