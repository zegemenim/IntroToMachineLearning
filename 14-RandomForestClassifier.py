import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("Datasets/14-income_evaluation.csv")
# https://www.kaggle.com/datasets/lodetomasi1995/income-classification

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

col_names = [
    "age",
    "workclass",
    "finalweight",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
df.columns = col_names

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

categorical_cols = [col for col in df.columns if df[col].dtype == "O"]
numeric_cols = [col for col in df.columns if df[col].dtype != "O"]

for col in categorical_cols:
    print(f"{col} value counts:\n{df[col].value_counts()}\n")

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.countplot(data=df, x="income", hue="sex")
# plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.countplot(data=df, x="income", hue="race")
# plt.show()

sns.barplot(data=df, y="hours_per_week", hue="income")
# plt.show()
print(df.workclass.unique())
df.workclass = df.workclass.replace(" ?", np.nan)

print(df.education.unique())
print(df.marital_status.unique())
print(df.occupation.unique())
df.occupation = df.occupation.replace(" ?", np.nan)
print(df.native_country.unique())
df.native_country = df.native_country.replace(" ?", np.nan)

# sns.pairplot(df, hue="income")
# plt.show()

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "O"]
numeric_cols = [col for col in X_train.columns if X_train[col].dtype != "O"]

print(X_train[categorical_cols].isnull().sum())
print(X_test[categorical_cols].isnull().sum())

for i in [X_train, X_test]:
    i["workclass"].fillna(X_train["workclass"].mode()[0], inplace=True)
    i["occupation"].fillna(X_train["occupation"].mode()[0], inplace=True)
    i["native_country"].fillna(X_train["native_country"].mode()[0], inplace=True)

y_train_binary = y_train.apply(lambda x: 1 if x.strip() == ">50K" else 0)
target_means = y_train_binary.groupby(X_train["native_country"]).mean()

X_train["native_country_encoded"] = X_train["native_country"].map(target_means)
X_train["native_country_encoded"].fillna(y_train_binary.mean(), inplace=True)

X_test["native_country_encoded"] = X_test["native_country"].map(target_means)
X_test["native_country_encoded"].fillna(y_train_binary.mean(), inplace=True)

X_train = X_train.drop("native_country", axis=1)
X_test = X_test.drop("native_country", axis=1)

one_hot_categories = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            one_hot_categories,
        )
    ],
    remainder="passthrough",
)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

col_names = encoder.get_feature_names_out()

X_train = pd.DataFrame(X_train_encoded, columns=col_names, index=X_train.index)
X_test = pd.DataFrame(X_test_encoded, columns=col_names, index=X_test.index)

# ---
# Not necessary
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
# ---

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, random_state=15)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

rfc = RandomForestClassifier(n_estimators=100, random_state=15)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print(rfc.feature_importances_)
feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(
    ascending=False
)

print(feature_scores)
# Take the last 10 features
least_important_features = feature_scores.tail(10).index.tolist()
print(least_important_features)
X_train_reduced = X_train.drop(least_important_features, axis=1)
X_test_reduced = X_test.drop(least_important_features, axis=1)

# Train the model again
rfc = RandomForestClassifier(n_estimators=100, random_state=15)
rfc.fit(X_train_reduced, y_train)
y_pred = rfc.predict(X_test_reduced)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    "n_estimators": [50, 100, 200, 300, 500, 1000],
    "max_depth": [None, 5, 10, 20, 30, 40, 50],
    "max_features": [5, 6, 7, 8, "sqrt", "log2"],
    "min_samples_split": [2, 5, 10, 15, 20]
}
rfc = RandomForestClassifier(random_state=15)
rfcv = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist, cv=3, n_jobs=-1)
rfcv.fit(X_train, y_train)
print("Best parameters found: ", rfcv.best_params_)
y_pred = rfcv.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# 86% accuracy after tuning
rfc_tuned = RandomForestClassifier(
    n_estimators=100,
    max_depth=40,
    max_features=8,
    min_samples_split=20,
    random_state=15,
)
rfc_tuned.fit(X_train, y_train)
y_pred = rfc_tuned.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# 86% accuracy after tuning