"""
Grid Search Cross Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv("../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/Advertising.csv")

X = df.drop("sales", axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

base_elastic_net_model = ElasticNet()

# Define parameter grid
param_grid = {
    'alpha': [0.1, 1, 5, 10, 50, 100],
    'l1_ratio': [.1, .5, .7, .95, .99, 1],
}

grid_model = GridSearchCV(estimator=base_elastic_net_model,
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=2)

grid_model.fit(X_train, y_train)

results = pd.DataFrame(grid_model.cv_results_)
print("GridSearchCV results:")
print(results)

print("GridSearchCV Best Estimator:", grid_model.best_estimator_)
print("GridSearchCV Best Params:", grid_model.best_params_)

# Once satisfied predict with test data

y_pred = grid_model.predict(X_test)
RMSE = root_mean_squared_error(y_test, y_pred)

print("GridSearchCV RMSE final:", RMSE)
