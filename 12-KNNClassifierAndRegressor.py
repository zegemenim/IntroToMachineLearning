import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Datasets/12-health_risk_classification.csv")
print(df.head())
print(df.info())
print(df.describe())

sns.scatterplot(x=df.blood_pressure_variation, y=df.activity_level_index, hue=df.high_risk_flag)
# plt.show()

X = df.drop("high_risk_flag", axis=1)
y = df["high_risk_flag"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
# weights can be 'uniform' or 'distance'
# uniform: all points in each neighborhood are weighted equally.
# distance: closer neighbors of a query point will have a greater influence than neighbors which are further away.
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred))

df_reg = pd.read_csv("Datasets/12-house_energy_regression.csv")
print(df_reg.head())
print(df_reg.info())
print(df_reg.describe())

sns.scatterplot(x=df_reg.outdoor_humidity_level, y=df_reg.daily_energy_consumption_kwh)
# plt.show()

sns.scatterplot(x=df_reg.avg_indoor_temp_change, y=df_reg.daily_energy_consumption_kwh)
# plt.show()

X_reg = df_reg.drop("daily_energy_consumption_kwh", axis=1)
y_reg = df_reg["daily_energy_consumption_kwh"]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=15
)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5, algorithm='auto')
knn_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_reg = knn_reg.predict(X_test_reg_scaled)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Regression Metrics:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R^2: {r2:.2f}")
plt.scatter(y_test_reg, y_pred_reg)
# plt.show()
