import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#https://www.kaggle.com/datasets/nsrose7224/crowdedness-at-the-campus-gym/data
df = pd.read_csv("Datasets/15-gym_crowdedness.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.date = pd.to_datetime(df['date'], utc=True)
df['year'] = df['date'].dt.year
df.drop('date', axis=1, inplace=True)
print(df.head())

# independent and dependent variables
X = df.drop('number_people', axis=1)
y = df['number_people']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def calculate_module_metrics(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    return mae, mse, rmse, r2

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "KNN Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
}

for i in range(len(models)):
    model = list(models.values())[i]
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = calculate_module_metrics(y_train, y_train_pred)
    model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = calculate_module_metrics(y_test, y_test_pred)

    print(f"{list(models.keys())[i]} Performance:")
    print(f"Train MAE: {model_train_mae}, MSE: {model_train_mse}, RMSE: {model_train_rmse}, R2: {model_train_r2}")
    print(f"Test MAE: {model_test_mae}, MSE: {model_test_mse}, RMSE: {model_test_rmse}, R2: {model_test_r2}")
    print("--------------------------------------------------")

# Multi hyperparameter tuning

knn_params = {
    'n_neighbors': [2, 3, 5, 7, 10, 20, 30, 40, 50, 100],
}

rf_params = {
    'n_estimators': [50, 100, 200, 300, 500, 1000],
    'max_depth': [None, 5, 8, 10, 15, 20],
    'max_features': [2, 3, 5, 7, 10, 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10, 15, 20],
}

from sklearn.model_selection import RandomizedSearchCV

randomcv_models = [
    ("KNN", KNeighborsRegressor(), knn_params),
    ("RF", RandomForestRegressor(), rf_params)
]

for name, model, params in randomcv_models:
    randomcv = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=100,
        cv=3,
        n_jobs=-1,
    )
    randomcv.fit(X_train_scaled, y_train)

    best_model = randomcv.best_estimator_
    print(f"{name} Best Model after Hyperparameter Tuning:")
    print(best_model)

models = {
    "KNN Regressor": KNeighborsRegressor(n_neighbors=2),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        max_features=7,
        random_state=15
    ),
}

for i in range(len(models)):
    model = list(models.values())[i]
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = calculate_module_metrics(y_train, y_train_pred)
    model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = calculate_module_metrics(y_test, y_test_pred)

    print(f"{list(models.keys())[i]} Performance:")
    print(f"Train MAE: {model_train_mae}, MSE: {model_train_mse}, RMSE: {model_train_rmse}, R2: {model_train_r2}")
    print(f"Test MAE: {model_test_mae}, MSE: {model_test_mse}, RMSE: {model_test_rmse}, R2: {model_test_r2}")
    print("--------------------------------------------------")

# Random Forest Regressor Performance:
# Train MAE: 1.4830166929072757, MSE: 4.912921405839121, RMSE: 2.2165110885892543, R2: 0.9904192113907996
# Test MAE: 4.0671228290235435, MSE: 37.30112894953506, RMSE: 6.107465018281731, R2: 0.9283573706317149