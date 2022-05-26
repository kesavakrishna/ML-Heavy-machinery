import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('bpp_training_data.csv')
del dataset['SalesID']
del dataset['MachineID']
del dataset['ModelID']
del dataset['datasource']
del dataset['auctioneerID']

bpp = pd.read_csv('bpp_test.csv')
del bpp['SalesID']
del bpp['MachineID']
del bpp['ModelID']
del bpp['datasource']
del bpp['auctioneerID']

y = dataset['SalePrice']
x = dataset.drop('SalePrice',axis = 1)

dataset.head

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)


#Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
acc = regressor.score(X_test, y_test)
print(acc)

y_pred = regressor.predict(bpp)

# df = pd.Series({'Real Values':y_test, 'Predicted Values':y_pred})


# print(df)

csv = pd.Series(y_pred)
csv.to_csv("pred.csv", index=False)