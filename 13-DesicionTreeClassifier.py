# https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Datasets/13-car_evaluation.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df.columns = col_names

for col in df.columns:
    print(f"{col} value counts:\n{df[col].value_counts()}\n")


df.doors = df.doors.replace("5more", 5).astype(int)

df.persons = df.persons.replace("more", 5).astype(int)

X = df.drop("class", axis=1)
y = df["class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_cols = ["buying", "maint", "lug_boot", "safety"]

numeric_cols = ["doors", "persons"]

ordinal_encoder = OrdinalEncoder(
    categories=[
        ["low", "med", "high", "vhigh"],  # buying
        ["low", "med", "high", "vhigh"],  # maint
        ["small", "med", "big"],  # lug_boot
        ["low", "med", "high"],  # safety
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("transformation_name_doesnt_matter", ordinal_encoder, categorical_cols)
    ],
    remainder="passthrough",
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)

tree_model.fit(X_train_transformed, y_train)
y_pred = tree_model.predict(X_test_transformed)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(12, 8))
from sklearn import tree

tree.plot_tree(tree_model.fit(X_train_transformed, y_train), feature_names=X.columns)
plt.show()

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [2, 3, 4, 5, 6, 15, None],
    "max_features": [None, "sqrt", "log2"],
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
)
import warnings

warnings.filterwarnings("ignore")
grid_search.fit(X_train_transformed, y_train)

print("Best parameters found: ", grid_search.best_params_)

y_pred_best = grid_search.best_estimator_.predict(X_test_transformed)
print("Accuracy after tuning:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix after tuning:\n", confusion_matrix(y_test, y_pred_best))
print(
    "Classification Report after tuning:\n", classification_report(y_test, y_pred_best)
)


tree_model_new = DecisionTreeClassifier(
    criterion="entropy", max_depth=None, max_features=None, splitter="best"
)

tree_model_new.fit(X_train_transformed, y_train)
y_pred_new = tree_model_new.predict(X_test_transformed)


print("Accuracy of new model:", accuracy_score(y_test, y_pred_new))
print("Confusion Matrix of new model:\n", confusion_matrix(y_test, y_pred_new))
print(
    "Classification Report of new model:\n", classification_report(y_test, y_pred_new)
)

plt.figure(figsize=(12, 8))
tree.plot_tree(
    tree_model_new.fit(X_train_transformed, y_train), feature_names=X.columns
)
plt.show()

df_new = pd.read_csv("Datasets/11-iris.csv")
print(df_new.head())
print(df_new.info())
print(df_new.describe())

X_new = df_new.drop(["Id", "Species"], axis=1)
y_new = df_new["Species"]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=10
)

tree_model_iris = DecisionTreeClassifier()
tree_model_iris.fit(X_train_new, y_train_new)
tree.plot_tree(
    tree_model_iris.fit(X_train_new, y_train_new),
    feature_names=X_new.columns,
    filled=True,
)
plt.show()

y_pred_iris = tree_model_iris.predict(X_test_new)
print("Iris Dataset Accuracy:", accuracy_score(y_test_new, y_pred_iris))
print("Iris Dataset Confusion Matrix:\n", confusion_matrix(y_test_new, y_pred_iris))
print(
    "Iris Dataset Classification Report:\n",
    classification_report(y_test_new, y_pred_iris),
)
