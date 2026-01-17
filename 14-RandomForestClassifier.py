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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)


categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "O"]
numeric_cols = [col for col in X_train.columns if X_train[col].dtype != "O"]

print(X_train[categorical_cols].isnull().sum())
print(X_test[categorical_cols].isnull().sum())

for i in [X_train, X_test]:
    i['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    i['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    i['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)

y_train_binary = y_train.apply(lambda x: 1 if x.strip() == ">50K" else 0)
target_means = y_train_binary.groupby(X_train['native_country']).mean()

X_train['native_country_encoded'] = X_train['native_country'].map(target_means)
X_train['native_country_encoded'].fillna(y_train_binary.mean(), inplace=True)

X_test['native_country_encoded'] = X_test['native_country'].map(target_means)
X_test['native_country_encoded'].fillna(y_train_binary.mean(), inplace=True)

X_train = X_train.drop('native_country', axis=1)
X_test = X_test.drop('native_country', axis=1)

one_hot_categories = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex'
]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), one_hot_categories)
    ],
    remainder="passthrough",
)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

col_names = encoder.get_feature_names_out()

X_train = pd.DataFrame(X_train_encoded, columns=col_names, index=X_train.index)
X_test = pd.DataFrame(X_test_encoded, columns=col_names, index=X_test.index)

#---
# Not necessary
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
#---
