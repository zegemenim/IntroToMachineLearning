import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Datasets/16-diabetes.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df.Insulin.value_counts())
print(df.columns)

columns_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for column in columns_to_check:
    zero_count = (df[column] == 0).sum()
    print(f"Number of zeros in {column}: {zero_count}")
    zero_percentage = (zero_count / len(df)) * 100
    print(f"Percentage of zeros in {column}: {zero_percentage:.2f}%")

sns.histplot(df["DiabetesPedigreeFunction"], bins=30, kde=True)
plt.title("Diabetes Pedigree Function Distribution")
plt.xlabel("Diabetes Pedigree Function")
plt.ylabel("Frekans")
plt.show()

sns.boxplot(x="Outcome", y="DiabetesPedigreeFunction", data=df)
plt.title("Outcome by Diabetes Pedigree Function")
plt.xlabel("Outcome (0=No-Diabetes, 1=Diabetes)")
plt.ylabel("Diabetes Pedigree Function")
plt.show()

sns.stripplot(
    x="Outcome", y="DiabetesPedigreeFunction", data=df, jitter=True, alpha=0.6
)
plt.title("Pedigree Values (by Outcome)")
plt.show()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

columns_to_fill = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


medians = {}

for column in columns_to_fill:
    medians_value = X_train[column].replace(0, np.nan).median()
    medians[column] = medians_value
    X_train[column] = X_train[column].replace(0, medians_value)

for column in columns_to_fill:
    X_test[column] = X_test[column].replace(0, medians[column])

print(X_train.describe())
print(X_test.describe())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ada = AdaBoostClassifier()
ada.fit(X_train_scaled, y_train)
y_pred = ada.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "learning_rate": [0.001, 0.01, 0.1, 1.0],
}

grid = GridSearchCV(estimator= AdaBoostClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)
print("Best Parameters:", grid.best_params_)

best_ada = AdaBoostClassifier(learning_rate=1, n_estimators=50)
best_ada.fit(X_train_scaled, y_train)
y_pred_best = best_ada.predict(X_test_scaled)
print("Confusion Matrix (Tuned):")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification Report (Tuned):")
print(classification_report(y_test, y_pred_best))
print("Accuracy (Tuned):", accuracy_score(y_test, y_pred_best))