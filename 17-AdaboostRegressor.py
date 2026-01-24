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

