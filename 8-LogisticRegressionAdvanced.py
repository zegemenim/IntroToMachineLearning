import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Datasets/8-fraud_detection.csv")
print(df.head())
print(df.info())
print(df.describe())

df.is_fraud.value_counts()
# Imbalanced dataset

# Preprocess data
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]
sns.scatterplot(x=X.transaction_amount, y=X.transaction_risk_score, hue=y)
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

penalties = ["l1", "l2", "elasticnet", "none"]
c_values = [0.01, 0.1, 1, 10, 100]
solvers = ["liblinear", "saga", "lbfgs", "newton-cg", "sag", "newton-cholesky"]
class_weights = [{0: w, 1: y} for w in [1, 10, 50, 100] for y in [1, 10, 50, 100]]
params = {
    "penalty": penalties,
    "C": c_values,
    "solver": solvers,
    "class_weight": class_weights,
}
cv = StratifiedKFold()
grid = GridSearchCV(
    estimator=model, param_grid=params, cv=cv, n_jobs=-1, scoring="accuracy"
)
import warnings

warnings.filterwarnings("ignore")

grid.fit(X_train, y_train)
print(f"Best Parameters: {grid.best_params_}")
print(f"Best Cross-Validation Score: {grid.best_score_:.2f}")
y_pred_grid = grid.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)
print(f"Grid Search Accuracy: {accuracy_grid:.2f}")
conf_matrix_grid = confusion_matrix(y_test, y_pred_grid)
print("Grid Search Confusion Matrix:")
print(conf_matrix_grid)
class_report_grid = classification_report(y_test, y_pred_grid)
print("Grid Search Classification Report:")
print(class_report_grid)

# ROC and AUC Explanation:
# The ROC Curve is a graphical representation of a classification model’s performance across different threshold values. It plots the True Positive Rate (Recall) on the Y-axis against the False Positive Rate (1 - Specificity) on the X-axis.
# 	•	A model that perfectly distinguishes between classes has a curve that reaches the top-left corner.
# 	•	The closer the curve is to the top-left, the better the model.
# 	•	The area under the ROC curve (AUC) quantifies this performance:
# 	•	AUC = 1 → perfect classifier
# 	•	AUC = 0.5 → random guessing

# ROC is especially useful for imbalanced datasets, as it evaluates the model independent of class distribution or threshold.

model_prob = grid.predict_proba(X_test)
print(model_prob)
model_prob = model_prob[:, 1]  # Probabilities for the positive (fraud) class

from sklearn.metrics import roc_curve, roc_auc_score

model_auc = roc_auc_score(y_test, model_prob)
print(f"Model AUC: {model_auc:.2f}")
roc_curve0 = roc_curve(y_test, model_prob)
print(roc_curve0)  # fpr (false positive rate), tpr (true positive rate), thresholds
model_fpr, model_tpr, model_thresholds = roc_curve(y_test, model_prob)
plt.plot(model_fpr, model_tpr, label="Logistic Regression", marker=".")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(model_fpr, model_tpr, marker=".", label="Logistic Regression")

import numpy as np

for fpr, tpr, threshold in zip(model_fpr, model_tpr, model_thresholds):
    ax.annotate(f"{np.round(threshold, 2)}", (fpr, tpr), ha="center")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve with Thresholds")
ax.legend()
plt.show()

custom_threshold = 0.2
y_pred_custom = (model_prob >= custom_threshold).astype(int)
print(
    f"Custom Threshold ({custom_threshold}) Accuracy: {accuracy_score(y_test, y_pred_custom):.2f}"
)
print("Custom Threshold Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))
print("Custom Threshold Classification Report:")
print(classification_report(y_test, y_pred_custom))
