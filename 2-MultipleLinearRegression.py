import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Datasets/2-multiplegradesdataset.csv')
print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe(), "\n")

sns.pairplot(df)
plt.show()

plt.scatter(df['Study Hours'], df['Exam Score'], color='r')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Study Hours vs Exam Score')
plt.show()

sns.regplot(x='Study Hours', y='Exam Score', data=df) # regression line
plt.show()

# independent and dependent variables
X = df.iloc[:, :-1].values  # all rows, all columns except the last
y = df.iloc[:, -1].values   # all rows, only the last column

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
print("Model trained successfully.\n")

new_student = [[5, 7, 90, 4]] 
predicted_score = regression.predict(scaler.transform(new_student))
print(f"Predicted Exam Score for new student: {predicted_score[0]}\n")

# prediction with test data
y_pred = regression.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
score = r2_score(y_test, y_pred)
print(f"R^2 Score: {score}\n")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores")
plt.show()

residuals = y_test - y_pred
sns.displot(residuals, kind="kde")
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()