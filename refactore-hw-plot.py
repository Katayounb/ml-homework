import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_breast_cancer
from mpl_toolkits.mplot3d import Axes3D

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

print('----new col names, now have 31----')
col_name = np.append(col_name, 'diagnosis')
print(col_name)

print('----now we have the df_cancer----')
df_cancer = pd.DataFrame(cancer_data, columns=col_name)
print(df_cancer)

print('--------------')
print(df_cancer['diagnosis'])
os.makedirs('plots', exist_ok=True)

plt.plot(df_cancer['mean area'], color='red')
plt.title('mean area')
plt.xlabel('Index')
plt.ylabel('last col')
plt.savefig(f'plots/mean_area_by_index_plot.png', format='png')
plt.clf()

# Plotting histogram
plt.hist(df_cancer['worst area'], bins=3, color='g')
plt.title('Worst Area')
plt.xlabel('Worst Area')
plt.ylabel('Count')
plt.savefig(f'plots/worst-area.png', format='png')
plt.clf()


# Plotting scatterplot - mean area and worst area
plt.scatter(df_cancer['mean area'], df_cancer['worst area'], color='b')
plt.title('Mean Area vs worst Area')
plt.xlabel('Mean Area')
plt.ylabel('worst Area')
plt.savefig(f'plots/mean_worst.png', format='png')

# Try to adapt 3D plot with tridsurf from viz-homework
fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_trisurf(df_cancer['worst perimeter'], df_cancer['worst area'], df_cancer['worst concavity'], color='blue', antialiased=True)
axes.set_xlabel('worst perimeter')
axes.set_ylabel('worst area')
axes.set_zlabel('worst concavity')
plt.savefig('plots/3d-trisurf.png')

## Reach
columns_names = cancer.feature_names
y = cancer.target
X = cancer.data
#print(columns_names.shape)
# Splitting features and target datasets into: train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Training a Linear Regression model with fit()
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# This part is for myself to learn, Output of the training is a model: a + b*X0 + c*X1 + d*X2 ...
print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, columns_names)}")


