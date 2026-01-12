import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import datasets
from lazypredict.Supervised import LazyRegressor

diabetes = datasets.load_diabetes()

X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
X = X.astype(np.float32)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=15
# )

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
print("Training models...\n")
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
