import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from joblib import dump, load

df = pd.read_csv('../../../UNZIP_FOR_NOTEBOOKS_FINAL/08-Linear-Regression-Models/Advertising.csv')

X = df.drop('sales', axis=1)
y = df['sales']

"""
To test performances on a particular dataset
of different degree of polynomial regression
we do the following steps:

1. create different order polynomial
2. split polynomial in train/test split
3. fit on train
4. store/save rmse for both train and test set
5. plot results
"""

train_rmse_error = []
test_rmes_error = []

for d in range(1, 10):

    poly_converter = PolynomialFeatures(degree=d, include_bias=False)
    poly_features = poly_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_rmse = root_mean_squared_error(y_train, train_prediction)
    test_rmse = root_mean_squared_error(y_test, test_prediction)

    train_rmse_error.append(train_rmse)
    test_rmes_error.append(test_rmse)

#print(train_rmse_error)
#print(test_rmes_error)

"""
plt.plot(range(1, 6), train_rmse_error[:5], label="Train RMSE")
plt.plot(range(1, 6), test_rmes_error[:5], label="Test RMSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.legend()

plt.show()
"""


# Deploy model and converter
final_poly_converter = PolynomialFeatures(degree=3, include_bias=False)
final_model = LinearRegression()
full_converted_X = final_poly_converter.fit_transform(X)
final_model.fit(full_converted_X, y)

dump(final_model, 'final_poly_model.joblib')
dump(final_poly_converter, 'final_converter.joblib')

loaded_converter = load('final_converter.joblib')
loaded_model = load('final_poly_model.joblib')

new_campaign = [[149, 22, 12]]
new_campaign_converted = loaded_converter.fit_transform(new_campaign)
new_campaign_predicted_sale = loaded_model.predict(new_campaign_converted)
print(new_campaign_predicted_sale)
