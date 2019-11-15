from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

col_name = cancer.feature_names
y = cancer.target
X = cancer.data
#print(X)
#print(y.data)

# Splitting features and target datasets into: train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Using KN-Regressor:
# Test 1- Try to see accuracy for Train and Test for KN-Reg
from sklearn.neighbors import KNeighborsRegressor
kn_reg = KNeighborsRegressor(n_neighbors=5, metric='minkowski', p=2)
kn_reg.fit(X_train, y_train)
print(f"KNeighborsRegressor, n-5, for test set: {kn_reg.score(X_test, y_test):.4f}")
print(f"KNeighborsRegressor, n-5, for train set: {kn_reg.score(X_train, y_train):.4f}")

# Using KN-classifier: n_neighbors default number is 5, metric default value is minkowski,
# auto: it automatically decide the best algorithm according to values of training data.fit()
# Test 2- Try to see accuracy for Train and Test for KN-Clf
# the result from test 1 (KN-Reg) vs Test 2 (KN-Clf) shows KN-Clf has better accuracy
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, algorithm='auto')
classifier.fit(X_train, y_train)
print(f"KNeighborsClassifier- accuracy, n-5, for test set: {classifier.score(X_test, y_test):.4f}")
print(f"KNeighborsClassifier- accuracy, n-5, for train set: {classifier.score(X_train, y_train):.4f}")