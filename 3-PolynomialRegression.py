import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("Datasets/3-customersatisfaction.csv")
print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe(), "\n")
df.drop("Unnamed: 0", axis=1, inplace=True)

sns.pairplot(df)
plt.show()

plt.scatter(df["Customer Satisfaction"], df["Incentive"], color="g")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
plt.title("Customer Satisfaction vs Incentive")
plt.show()

# independent and dependent variables
X = df[["Customer Satisfaction"]]
y = df["Incentive"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15
)

# scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regression = LinearRegression()

regression.fit(X_train, y_train)
print("Model trained successfully.\n")

# Prediction
y_pred = regression.predict(X_test)

score = r2_score(y_test, y_pred)
print(f"R^2 Score: {score}\n")

plt.scatter(X_train, y_train, color="blue")
plt.plot(X_train, regression.predict(X_train), color="red")
plt.show()

# Polynomial Regression
# poly = PolynomialFeatures(degree=3)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_regression = LinearRegression()
poly_regression.fit(X_train_poly, y_train)
print("Polynomial Model trained successfully.\n")

y_poly_pred = poly_regression.predict(X_test_poly)
poly_score = r2_score(y_test, y_poly_pred)
print(f"Polynomial R^2 Score: {poly_score}\n")

coefficients = poly_regression.coef_
intercept = poly_regression.intercept_
print(f"Polynomial Coefficients: {coefficients}")
print(f"Polynomial Intercept: {intercept}\n")

plt.scatter(X_train, y_train, color="blue")
plt.scatter(X_train, poly_regression.predict(X_train_poly), color="red")
plt.title("Polynomial Regression Fit")
plt.show()

new_df = pd.read_csv("Datasets/3-newdatas.csv")
new_df.rename(columns={"0": "Customer_Satisfaction"}, inplace=True)

new_X = new_df[["Customer_Satisfaction"]]
new_X = scaler.fit_transform(new_X)
new_X_poly = poly.transform(new_X)
new_y = poly_regression.predict(new_X_poly)
plt.plot(new_X, new_y, color="green", label="Predictions on New Data")
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="red", label="Test Data")
plt.title("Polynomial Regression Predictions on New Data")
plt.legend()
plt.show()

# Pipeline for Polynomial Regression


def poly_regression(degree=2):
    poly_features = PolynomialFeatures(degree=degree)
    lin_reg = LinearRegression()
    pipeline = Pipeline(
        [
            ("standard_scaler", StandardScaler()),
            ("poly_features", poly_features),
            ("lin_reg", lin_reg),
        ]
    )
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Polynomial Regression R^2 Score (degree={degree}): {score}")

    y_pred_new = pipeline.predict(new_X)
    plt.plot(
        new_X,
        y_pred_new,
        color="purple",
        label=f"Polynomial Regression Predictions (degree={degree})",
    )
    plt.scatter(X_train, y_train, color="blue", label="Training Data")
    plt.scatter(X_test, y_test, color="red", label="Test Data")
    plt.title(f"Polynomial Regression Predictions on New Data (degree={degree}) r2={score:.4f}")
    plt.legend()
    plt.show()

    return pipeline, score


poly_regression(2)
poly_regression(3)
poly_regression(4)
poly_regression(5)