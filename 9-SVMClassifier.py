import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Datasets/9-email_classification_svm.csv")
print(df.head())
print(df.info())
print(df.describe())
# subject_formality_score -> sender formality score
# sender_relationship_score -> relationship score with sender
# email_type -> 0: Personal, 1: Work
print(df.email_type.value_counts())
# Balanced dataset

sns.scatterplot(
    x=df.subject_formality_score,
    y=df.sender_relationship_score,
    hue=df.email_type,
)
# plt.show()

X = df.drop("email_type", axis=1)
y = df["email_type"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15
)
from sklearn.svm import SVC

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("---- Tuning SVM with RBF Kernel ----")
rbf = SVC(kernel="rbf")
rbf.fit(X_train, y_train)
y_pred_rbf = rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"RBF Kernel Accuracy: {accuracy_rbf:.2f}")
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
print("RBF Kernel Confusion Matrix:")
print(conf_matrix_rbf)
print("RBF Kernel Classification Report:")
print(classification_report(y_test, y_pred_rbf))

df = pd.read_csv("Datasets/9-loan_risk_svm.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.loan_risk.value_counts())
# Balanced dataset

sns.scatterplot(
    x=df.credit_score_fluctuation,
    y=df.recent_transaction_volume,
    hue=df.loan_risk,
)
# plt.show()

X = df.drop("loan_risk", axis=1)
y = df["loan_risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15
)
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)
y_pred3 = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred3)
print(f"Accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred3)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred3))
print("---- Tuning SVM with RBF Kernel ----")
rbf = SVC(kernel="rbf")
rbf.fit(X_train, y_train)
y_pred_rbf4 = rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf4)
print(f"RBF Kernel Accuracy: {accuracy_rbf:.2f}")
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf4)
print("RBF Kernel Confusion Matrix:")
print(conf_matrix_rbf)
print("RBF Kernel Classification Report:")
print(classification_report(y_test, y_pred_rbf4))

print("---- SVM with Polynomial Kernel ----")
poly = SVC(kernel="poly", degree=3)
poly.fit(X_train, y_train)
y_pred_poly = poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Polynomial Kernel Accuracy: {accuracy_poly:.2f}")
conf_matrix_poly = confusion_matrix(y_test, y_pred_poly)
print("Polynomial Kernel Confusion Matrix:")
print(conf_matrix_poly)
print("Polynomial Kernel Classification Report:")
print(classification_report(y_test, y_pred_poly))

print("---- SVM with Sigmoid Kernel ----")
sigmoid = SVC(kernel="sigmoid")
sigmoid.fit(X_train, y_train)
y_pred_sigmoid = sigmoid.predict(X_test)
accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
print(f"Sigmoid Kernel Accuracy: {accuracy_sigmoid:.2f}")
conf_matrix_sigmoid = confusion_matrix(y_test, y_pred_sigmoid)
print("Sigmoid Kernel Confusion Matrix:")
print(conf_matrix_sigmoid)
print("Sigmoid Kernel Classification Report:")
print(classification_report(y_test, y_pred_sigmoid))

param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": ["scale", "auto"],
    "kernel": ["rbf", "poly", "sigmoid"],
}
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
)
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


df = pd.read_csv("Datasets/9-seismic_activity_svm.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.seismic_event_detected.value_counts())
# Balanced dataset

sns.scatterplot(
    x=df.underground_wave_energy,
    y=df.vibration_axis_variation,
    hue=df.seismic_event_detected,
)
# plt.show()

df["underground_wave_energy **2"] = df["underground_wave_energy"] ** 2
df["vibration_axis_variation **2"] = df["vibration_axis_variation"] ** 2
df["energy_variation_interaction"] = (
    df["underground_wave_energy"] * df["vibration_axis_variation"]
)

X = df.drop("seismic_event_detected", axis=1)
y = df["seismic_event_detected"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15
)

import plotly.express as px

fig = px.scatter_3d(
    df,
    x="underground_wave_energy **2",
    y="vibration_axis_variation **2",
    z="energy_variation_interaction",
    color=df.seismic_event_detected.astype(str),
    title="3D Scatter Plot of Seismic Data",
)
# fig.show()

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)
y_pred5 = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred5)
print(f"Accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred5)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred5))

print("---- Tuning SVM with RBF Kernel ----")
df = pd.read_csv("Datasets/9-seismic_activity_svm.csv")

X = df.drop("seismic_event_detected", axis=1)
y = df["seismic_event_detected"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15
)
rbf = SVC(kernel="rbf")
rbf.fit(X_train, y_train)
y_pred_rbf6 = rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf6)
print(f"RBF Kernel Accuracy: {accuracy_rbf:.2f}")
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf6)
print("RBF Kernel Confusion Matrix:")
print(conf_matrix_rbf)
print("RBF Kernel Classification Report:")
print(classification_report(y_test, y_pred_rbf6))
