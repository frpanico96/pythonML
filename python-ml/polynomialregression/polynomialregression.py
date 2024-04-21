"""
Polynomial regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('../../UNZIP_FOR_NOTEBOOKS_FINAL/08-Linear-Regression-Models/Advertising.csv')

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

