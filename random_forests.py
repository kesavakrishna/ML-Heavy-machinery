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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, criterion="squared_error", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0, max_samples=None )
regressor.fit(X_train, y_train)
acc = regressor.score(X_test, y_test)
print(acc)

y_pred = regressor.predict(bpp)


# # df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
# print(df)

# csv = pd.Series(y_pred)
# csv.to_csv("pred.csv", index=False)


