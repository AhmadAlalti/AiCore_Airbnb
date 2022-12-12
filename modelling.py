import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

df_listing = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')

X, y = load_airbnb(df_listing, "Price_Night")
X = X.select_dtypes(include="number")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1) #Is the test size correct? original 30%, split into two 15%

reg = SGDRegressor()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)

MSE = mean_squared_error(y_train, y_train_pred)

print(f"Loss =  {MSE}")

