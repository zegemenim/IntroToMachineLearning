import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('1-studyhours.csv')
print(df.head(), "\n")
print(df.info(), "\n")

# Statistical Summary
print(df.describe(), "\n")
plt.scatter(df['Study Hours'], df['Exam Score'], color='blue')
plt.title('Study Hours vs Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.show()

# independent and dependent variables
X = df[['Study Hours']]
y = df['Exam Score']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Standardize the data set (It's so important to fix outlier problems)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # to eliminate significant value differences between columns
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # fit -> data leakage

# Training the model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
print("Model trained successfully.\n")
print(f"Coefficient: {regression.coef_[0]}")
print(f"Intercept: {regression.intercept_}\n")
plt.scatter(X_train, y_train)
plt.plot(X_train, regression.predict(X_train), 'r')
plt.show()
#predict
print(regression.predict(scaler.transform([[10]])))
print(regression.predict(scaler.transform([[20]])))
print(regression.predict(scaler.transform([[30]])))
print(regression.predict(scaler.transform([[40]])))
print(regression.predict(scaler.transform([[50]])))
print(regression.predict(scaler.transform([[60]])))
print(regression.predict(scaler.transform([[70]])))

# prediction with test data
y_pred = regression.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")
print('Adjusted R^2 Score:', 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)) # adjusted R^2 score calculation !!!
