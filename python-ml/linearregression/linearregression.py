import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Reading data

df = pd.read_csv('../../UNZIP_FOR_NOTEBOOKS_FINAL/08-Linear-Regression-Models/Advertising.csv')

# print(df.head())

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

axes[0].plot(df['TV'], df['sales'], 'o')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'], df['sales'], 'o')
axes[1].set_ylabel("Sales")
axes[1].set_title("Radio Spend")

axes[2].plot(df['newspaper'], df['sales'], 'o')
axes[2].set_ylabel("Sales")
axes[2].set_title("Newspaper Spend")

# plt.show()

'''
Drop non-feature columns to get X component
and define y component
and split into 4 components via the scikit-learn library

"test_size" parameter indicates which percentage of the data should be in the test set
"random_state" set a random seed
'''

X = df.drop('sales', axis=1)
y = df['sales']

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=101)
