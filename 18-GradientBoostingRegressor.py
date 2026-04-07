import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Load dataset
data = pd.read_csv("Datasets/18-concrete_data.csv")
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

print(data.corr())
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
# plt.show()

X = data.drop("Strength", axis=1)
y = data["Strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

tree_reg1 = DecisionTreeRegressor(max_depth=3)
tree_reg1.fit(X_train, y_train)
y2 = y_train - tree_reg1.predict(X_train)
print(y2)

tree_reg2 = DecisionTreeRegressor(max_depth=4)
tree_reg2.fit(X_train, y2)
y3 = y2 - tree_reg2.predict(X_train)
print(y3)

tree_reg3 = DecisionTreeRegressor(max_depth=5)
tree_reg3.fit(X_train, y3)
y4 = y3 - tree_reg3.predict(X_train)
print(y4)

y_pred = sum(tree.predict(X_train) for tree in [tree_reg1, tree_reg2, tree_reg3])
print(r2_score(y_train, y_pred))

tree_reg4 = DecisionTreeRegressor(max_depth=4)
tree_reg4.fit(X_train, y4)
y_pred = sum(
    tree.predict(X_train) for tree in [tree_reg1, tree_reg2, tree_reg3, tree_reg4]
)
print(r2_score(y_train, y_pred))

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_train)
print(r2_score(y_train, y_pred_gbr))

# Hyperparameter Tuning
params = {
    "n_estimators": [50, 100, 150, 200],
    "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0],
    "max_depth": [3, 4, 5],
    "loss": ["squared_error", "huber", "absolute_error", "quantile"],
}

from sklearn.model_selection import RandomizedSearchCV
rscv = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(), param_distributions=params, scoring="r2", cv=5
)
rscv.fit(X_train, y_train)
print("Best Parameters:", rscv.best_params_)
y_pred_gbr_tuned = rscv.best_estimator_.predict(X_train)
print("R^2 Score (Tuned):", r2_score(y_train, y_pred_gbr_tuned))