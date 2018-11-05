import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Normalizer

df = pd.read_csv('Microsoft.csv')
X_train = df[['open', 'high', 'low', 'volume', 'Close_Nasdaq', 'Volume_Nasdaq', 'UFRI_rate', 'DollarEuro']]
pd.scatter_matrix(X_train.loc[:, 'open':'DollarEuro'],  diagonal='hist', color='b')
plt.show()
y_train = df[['close']]
df2 = pd.read_csv('MSFT_18.csv')
X_test = df2[['open', 'high', 'low', 'volume', 'Close_Nasdaq', 'Volume_Nasdaq', 'UFRI_rate', 'DollarEuro']]
# So we need to eradicate all NaN values and replace them with some suitable values
# X_test['Volume_Nasdaq'].fillna(X_test['Volume_Nasdaq'].median(), inplace=True)
# X_test['Volume_Nasdaq'].fillna(X_test[''].median(), inplace=True)
print(X_test.isna().any())          # gives True
print(X_train.isna().any())         # gives True
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
print(X_test.isna().any())          # gives True
print(X_train.isna().any())         # remove this afterwards4
np.where(X_train.values >= np.finfo(np.float64).max)
np.where(X_test.values >= np.finfo(np.float64).max)

y_test = df2[['close']]
model = KNeighborsRegressor().fit(X_train, y_train)
print(model.score(X_test, y_test))
