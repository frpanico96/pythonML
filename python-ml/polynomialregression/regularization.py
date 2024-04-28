"""
Data Regularization and Ridge Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics._scorer import _SCORERS

df = pd.read_csv('../../UNZIP_FOR_NOTEBOOKS_FINAL/08-Linear-Regression-Models/Advertising.csv')

X = df.drop('sales', axis=1)
y = df['sales']

"""
Create interaction matrix
and perform train|test split
"""

polynomial_converter = PolynomialFeatures(degree=3, include_bias=False)

poly_feature = polynomial_converter.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(poly_feature, y, test_size=0.3, random_state=101)

"""
Scale data
Scaling fit should be done only on the training set
Since we don't want to infer anything on the test set
Transformation of data can be performed on both sets separately
"""
scaler = StandardScaler()
scaler.fit(X_train)

scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

"""
Ridge regression
"""

"""
To find the best alpha parameter cross-validation needs to be performed
To achieve it
import RidgeCV from sklearn.linear_model

Using ridgeCv a portion of the train set will be used as validation set
"""

ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10), scoring='neg_mean_absolute_error')
ridge_cv_model.fit(scaled_X_train, y_train)
print(ridge_cv_model.alpha_)
best_alpha = ridge_cv_model.alpha_
# print(_SCORERS.keys())

ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(scaled_X_train, y_train)
test_prediction = ridge_model.predict(scaled_X_test)

MAE = mean_absolute_error(y_test, test_prediction)
RMSE = root_mean_squared_error(y_test, test_prediction)

print(MAE, RMSE)


