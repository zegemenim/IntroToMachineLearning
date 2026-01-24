import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Datasets/17-cardekho.csv")
print(df.head())

df = df.drop("Unnamed: 0", axis=1)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df['seats'].value_counts()
print(df[df.duplicated()])

df = df.drop_duplicates(keep="first", ignore_index = True)
print(df['seats'].value_counts())

print(df[df['seats'] == 0])

df.loc[df['seats'] == 0, "seats"] = 5
print(df[df['seats'] == 0])

print(df['seats'].value_counts())
print(df.describe())

pd.set_option('display.float_format', '{:.2f}'.format)
print(df.describe())

sns.scatterplot(df, x="vehicle_age", y="selling_price", hue="fuel_type")
plt.title("Vehicle Age vs Selling Price by Fuel Type")
plt.xlabel("Vehicle Age (years)")
plt.ylabel("Selling Price")
# plt.show()

df = df[df.selling_price < 15000000]
print(df.describe())

sns.boxplot(x="fuel_type", y="selling_price", data=df)
plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
# plt.show()

sns.boxplot(x="transmission", y="selling_price", data=df)
plt.title("Selling Price by Transmission Type")
plt.xlabel("Transmission Type")
plt.ylabel("Selling Price")
# plt.show()

sns.scatterplot(x="km_driven", y="selling_price", data=df)
plt.title("KM Driven vs Selling Price")
plt.xlabel("KM Driven")
plt.ylabel("Selling Price")
# plt.show()

df = df[df.km_driven < 1000000]
print(df.describe())

sns.boxplot(x="seats", y="selling_price", data=df)
plt.title("Selling Price by Number of Seats")
plt.xlabel("Number of Seats")
plt.ylabel("Selling Price")
# plt.show()

print(df.corr())

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

cat_cols = df.select_dtypes("object").columns.tolist()
print(cat_cols)
unique_values = df[cat_cols].nunique()
print(unique_values)

# seller_type, fuel_type, transmission - one-hot encoding
# car_name, brand, model - frequency encoding
one_hot_cols = ["seller_type", "fuel_type", "transmission"]
freq_encode_cols = ["car_name", "brand", "model"]
for col in freq_encode_cols:
    freq = X_train[col].value_counts() / len(X_train)
    X_train[col + "_freq"] = X_train[col].map(freq)
    X_test[col + "_freq"] = X_test[col].map(freq)

    mean_freq = freq.mean()
    X_test[col + "_freq"] = X_test[col + "_freq"].fillna(mean_freq)

print(X_train.head())

X_train = X_train.drop(["car_name", "brand", "model"], axis=1)
X_test = X_test.drop(["car_name", "brand", "model"], axis=1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), one_hot_cols)
    ], remainder="passthrough"
)

X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

encoded_cols = transformer.get_feature_names_out()
print(encoded_cols)

X_train = pd.DataFrame(X_train, columns=encoded_cols)
X_test = pd.DataFrame(X_test, columns=encoded_cols)
print(X_train.head())
print(X_test.head())

