import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Reading data

df = pd.read_csv('../../UNZIP_FOR_NOTEBOOKS_FINAL/08-Linear-Regression-Models/Advertising.csv')

# print(df.head())

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
#
# axes[0].plot(df['TV'], df['sales'], 'o')
# axes[0].set_ylabel("Sales")
# axes[0].set_title("TV Spend")
#
# axes[1].plot(df['radio'], df['sales'], 'o')
# axes[1].set_ylabel("Sales")
# axes[1].set_title("Radio Spend")
#
# axes[2].plot(df['newspaper'], df['sales'], 'o')
# axes[2].set_ylabel("Sales")
# axes[2].set_title("Newspaper Spend")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
# help(LinearRegression)

'''
Create an instance of the model
Fit with train components
Predict model with X_test component
'''

model = LinearRegression()
model.fit(X_train, y_train)
test_prediction = model.predict(X_test)
# print(test_prediction)


'''
Evaluate the prediction with different metrics
'''
mae = mean_absolute_error(y_true=y_test, y_pred=test_prediction)
print(mae)
root_mse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_prediction))
print(root_mse)

'''
Residual analysis
'''

test_residuals = y_test - test_prediction
# sns.scatterplot(x=y_test, y=test_residuals)
# plt.axhline(y=0, color='red', ls='--')


# sns.displot(test_residuals, bins=25, kde=True)


'''
Sciplot allows to plot residuals distribution against normal distribution
'''

fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
_ = sp.stats.probplot(test_residuals, plot=ax)

plt.show()
