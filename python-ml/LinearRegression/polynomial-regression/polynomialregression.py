"""
Polynomial regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('../../../UNZIP_FOR_NOTEBOOKS_FINAL/08-Linear-Regression-Models/Advertising.csv')

X = df.drop('sales', axis=1)
y = df['sales']

"""
Calculate interaction matrix
with PolynomialFeatures
"""

polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
polynomial_converter.fit(X)
transformed_data = polynomial_converter.transform(X)

print(transformed_data)
print(transformed_data.shape)
print(transformed_data[0])


"""
Train test split
The feature dataset is not X but the interaction matrix

And fit the model with 9 features (interaction matrix) instead of 3
"""

X_train, X_test, y_train, y_test = train_test_split(transformed_data, y, test_size=0.3, random_state=101)
model = LinearRegression()

model.fit(X_train, y_train)


"""
Predict the model
"""

test_prediction = model.predict(X_test)
MAE = mean_absolute_error(y_test, test_prediction)
MSE = np.sqrt(mean_squared_error(y_test, test_prediction))

print("Mean Absolute Error:", MAE)
print("Mean Square Error:", MSE)
print("Coefficients", model.coef_)

