import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Load dataset
df = pd.read_csv('Datasets/6-bank_customers.csv')
print(df.head())
print(df.info())
print(df.describe())
# Preprocess data
X = df.drop('subscribed', axis=1)
y = df['subscribed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

model = LogisticRegression()
penalties = ['l1', 'l2', 'elasticnet', 'none']
c_values = [0.01, 0.1, 1, 10, 100]
solvers = ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag', 'newton-cholesky']
params = {'penalty': penalties, 'C': c_values, 'solver': solvers}

from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold()
grid = GridSearchCV(estimator=model, param_grid=params, cv=cv, n_jobs=-1, scoring='accuracy')
grid.fit(X_train, y_train)
print(f'Best Parameters: {grid.best_params_}')
print(f'Best Cross-Validation Score: {grid.best_score_:.2f}')
y_pred_grid = grid.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)
print(f'Grid Search Accuracy: {accuracy_grid:.2f}')
conf_matrix_grid = confusion_matrix(y_test, y_pred_grid)
print('Grid Search Confusion Matrix:')
print(conf_matrix_grid)
class_report_grid = classification_report(y_test, y_pred_grid)
print('Grid Search Classification Report:')
print(class_report_grid)

# Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV

model = LogisticRegression()
random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=10, cv=5, n_jobs=-1, scoring='accuracy')
random_search.fit(X_train, y_train)
print(f'Best Parameters (Randomized Search): {random_search.best_params_}')
print(f'Best Cross-Validation Score (Randomized Search): {random_search.best_score_:.2f}')
y_pred_random = random_search.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
print(f'Randomized Search Accuracy: {accuracy_random:.2f}')
conf_matrix_random = confusion_matrix(y_test, y_pred_random)
print('Randomized Search Confusion Matrix:')
print(conf_matrix_random)
class_report_random = classification_report(y_test, y_pred_random)
print('Randomized Search Classification Report:')
print(class_report_random)