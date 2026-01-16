import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Datasets/10-diamonds.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop("Unnamed: 0", axis=1)
print(df[df["x"] == 0])
df = df[df["x"] != 0]
print(df[df["y"] == 0])
print(df[df["z"] == 0])
df = df[df["z"] != 0]

sns.pairplot(df, diag_kind="kde")
# plt.show()

print(df.shape)

df = df[(df["depth"] < 75) & (df["depth"] > 45)]
df = df[(df["table"] < 75) & (df["table"] > 40)]
df = df[(df["z"] < 30) & (df["z"] > 2)]
df = df[(df["y"] < 20)]

print(df.shape)
print(df.cut.value_counts())
print(df.color.value_counts())
print(df.clarity.value_counts())

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=30
)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ["cut", "color", "clarity"]:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("---- Linear Regression Results ----")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

print("---- SVM Regressor Results ----")
from sklearn.svm import SVR

svr = SVR()
# svr.fit(X_train, y_train)
# y_pred_svr = svr.predict(X_test)
# mae_svr = mean_absolute_error(y_test, y_pred_svr)
# mse_svr = mean_squared_error(y_test, y_pred_svr)
# rmse_svr = np.sqrt(mse_svr)
# r2_svr = r2_score(y_test, y_pred_svr)
# print(f"Mean Absolute Error: {mae_svr:.2f}")
# print(f"Mean Squared Error: {mse_svr:.2f}")
# print(f"Root Mean Squared Error: {rmse_svr:.2f}")
# print(f"R^2 Score: {r2_svr:.2f}")

# Hyperparameter Tuning for SVR
print("---- Tuning SVR with RBF Kernel ----")
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf", "poly", "sigmoid", "linear"],
}

grid = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=3,
)
grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")
best_svr = grid.best_estimator_
y_pred_best = best_svr.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)
print("---- Best SVR Results ----")
print(f"Mean Absolute Error: {mae_best:.2f}")
print(f"Mean Squared Error: {mse_best:.2f}")
print(f"Root Mean Squared Error: {rmse_best:.2f}")
print(f"R^2 Score: {r2_best:.2f}")
