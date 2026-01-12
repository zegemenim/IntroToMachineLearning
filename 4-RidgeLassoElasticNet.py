import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Datasets/4-Algerian_forest_fires_dataset.csv")

print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe(), "\n")

print(df[df.isnull().any(axis=1)], "\n")  # Check for missing values
df.drop(122, inplace=True)  # Drop rows with missing values
df.loc[:123, "Region"] = 0
df.loc[123:, "Region"] = 1  # Encode categorical variable
df = df.dropna().reset_index(drop=True)
print(df.info(), "\n")
df.columns = df.columns.str.strip()
sns.pairplot(df)
plt.show()
print(df.day.unique())
# Delete if day = day
df.drop(df[df["day"] == "day"].index, inplace=True)
df = df.reset_index(drop=True)
print(df.info(), "\n")
# Convert first 6 columns to int
for col in df.columns[:6]:
    df[col] = df[col].astype(int)
print(df.info(), "\n")
# Convert columns to float between 6 and 12
for col in df.columns[6:13]:
    df[col] = df[col].astype(float)
print(df.info(), "\n")

print(df.Classes.unique())

# for fire = 1, no fire = 0
# print(df.loc[0]["Classes"], "\n!!!!")
df["Classes"] = df["Classes"].str.strip()
print(df["Classes"].unique())
df["Classes"] = df["Classes"].map({"fire": 1, "not fire": 0})
print(df.info(), "\n")
print(df.head(), "\n")
print(df.Classes.value_counts(), "\n")
df.drop(["day", "month", "year"], axis=1, inplace=True)

# independent and dependent variables
X = df.drop("FWI", axis=1)
y = df["FWI"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15
)

# Redundancy, multicollinearity, overfitting
print(X_train.corr(), "\n")


def correlation_for_dropping(dataframe, threshold):
    columnsToDrop = set()
    corr = dataframe.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) >= threshold:
                print(
                    f"Dropping column: {corr.columns[i]} due to high correlation with {corr.columns[j]}"
                )
                columnsToDrop.add(corr.columns[i])
    return columnsToDrop


cols_to_drop = correlation_for_dropping(X_train, 0.85)
X_train = X_train.drop(columns=cols_to_drop, axis=1)
X_test = X_test.drop(columns=cols_to_drop, axis=1)
print(X_train.corr(), "\n")

# scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=X_train)
plt.title("X_train")
plt.subplot(1, 2, 2)
sns.boxplot(data=X_train_scaled)
plt.title("X_train_scaled")
plt.show()

linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_pred_linear = linear.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_linear)
mse = mean_squared_error(y_test, y_pred_linear)
rmse = np.sqrt(mse)
score = r2_score(y_test, y_pred_linear)
print("Linear Regression Results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {score}\n")
plt.scatter(y_test, y_pred_linear)
plt.show()

# Test Lasso
lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_lasso)
mse = mean_squared_error(y_test, y_pred_lasso)
rmse = np.sqrt(mse)
score = r2_score(y_test, y_pred_lasso)
print("Lasso Regression Results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {score}\n")
plt.scatter(y_test, y_pred_lasso)
plt.show()

# Test Ridge
ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_ridge)
mse = mean_squared_error(y_test, y_pred_ridge)
rmse = np.sqrt(mse)
score = r2_score(y_test, y_pred_ridge)
print("Ridge Regression Results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {score}\n")
plt.scatter(y_test, y_pred_ridge)
plt.show()

# Test ElasticNet
elastic_net = ElasticNet()
elastic_net.fit(X_train_scaled, y_train)
y_pred_elastic = elastic_net.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_elastic)
mse = mean_squared_error(y_test, y_pred_elastic)
rmse = np.sqrt(mse)
score = r2_score(y_test, y_pred_elastic)
print("ElasticNet Regression Results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {score}\n")
plt.scatter(y_test, y_pred_elastic)
plt.show()

# LassoCV
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train_scaled, y_train)
y_pred_lasso_cv = lasso_cv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_lasso_cv)
mse = mean_squared_error(y_test, y_pred_lasso_cv)
rmse = np.sqrt(mse)
score = r2_score(y_test, y_pred_lasso_cv)
print("LassoCV Regression Results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {score}\n")
plt.scatter(y_test, y_pred_lasso_cv)
plt.show()
print(f"Optimal alpha for LassoCV: {lasso_cv.alpha_}\n")
print(f"LassoCV alphas: {lasso_cv.alphas_}\n")

# Ridge Cross-Validation
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(cv=5)
ridge_cv.fit(X_train_scaled, y_train)
y_pred_ridge_cv = ridge_cv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_ridge_cv)
mse = mean_squared_error(y_test, y_pred_ridge_cv)
rmse = np.sqrt(mse) 
score = r2_score(y_test, y_pred_ridge_cv)
print("RidgeCV Regression Results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {score}\n")
plt.scatter(y_test, y_pred_ridge_cv)
plt.show()
print(f"Optimal alpha for RidgeCV: {ridge_cv.alpha_}\n")

# ElasticNetCV
from sklearn.linear_model import ElasticNetCV
elastic_net_cv = ElasticNetCV(cv=5)
elastic_net_cv.fit(X_train_scaled, y_train)
y_pred_elastic_cv = elastic_net_cv.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_elastic_cv)
mse = mean_squared_error(y_test, y_pred_elastic_cv)
rmse = np.sqrt(mse)
score = r2_score(y_test, y_pred_elastic_cv)
print("ElasticNetCV Regression Results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {score}\n")
plt.scatter(y_test, y_pred_elastic_cv)
plt.show()
print(f"Optimal alpha for ElasticNetCV: {elastic_net_cv.alpha_}\n")
print(f"Optimal l1_ratio for ElasticNetCV: {elastic_net_cv.alphas_}\n")