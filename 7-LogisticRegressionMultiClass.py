import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# src_packet_rate -> Source-side packet transmission rate
# dst_packet_rate -> Destination-side packet reception rate
# avg_payload_size -> Average size of payload in packets
# connection_duration -> Duration of the connection (in seconds)
# tcp_flag_count -> Number of TCP flag occurrences
# avg_interarrival_time -> Time between packet arrivals
# failed_login_attempts -> Number of failed login attempts
# unusual_port_activity_score -> Score representing unusual port usage
# session_entropy -> Entropy of session behavior (for anomaly detection)
# avg_response_delay -> Average delay in server response (in ms)
# attack_type -> 0 = Normal, 1 = DDoS, 2 = Port Scan

df = pd.read_csv("Datasets/7-cyber_attack_data.csv")
print(df.head())
print(df.info())
print(df.describe())

# Preprocess data
X = df.drop("attack_type", axis=1)
y = df["attack_type"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
# Hyperparameter Tuning with Grid Search CV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

penalties = ["l1", "l2", "elasticnet", "none"]
c_values = [0.01, 0.1, 1, 10, 100]
solvers = ["liblinear", "saga", "lbfgs", "newton-cg", "sag", "newton-cholesky"]
params = {"penalty": penalties, "C": c_values, "solver": solvers}
cv = StratifiedKFold()
grid = GridSearchCV(
    estimator=model, param_grid=params, cv=cv, n_jobs=-1, scoring="accuracy"
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

# one vs one - one vs rest logistic regression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

ovo = OneVsOneClassifier(LogisticRegression())
ovr = OneVsRestClassifier(LogisticRegression())
ovo.fit(X_train, y_train)
ovr.fit(X_train, y_train)
y_pred_ovo = ovo.predict(X_test)
y_pred_ovr = ovr.predict(X_test)
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
print(f"One vs One Accuracy: {accuracy_ovo:.2f}")
print(f"One vs Rest Accuracy: {accuracy_ovr:.2f}")
conf_matrix_ovo = confusion_matrix(y_test, y_pred_ovo)
conf_matrix_ovr = confusion_matrix(y_test, y_pred_ovr)
print("One vs One Confusion Matrix:")
print(conf_matrix_ovo)
print("One vs Rest Confusion Matrix:")
print(conf_matrix_ovr)