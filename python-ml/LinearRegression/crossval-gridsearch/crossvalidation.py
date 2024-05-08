"""
Cross Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv("../../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/Advertising.csv")


"""
0. Clean and adjust data as necessary for X and Y
1. Split data in Train/Test for both X and y
2. Fit/Train Scaler on training X Data
3. Scale X test Data
4. Create Model
5. Fit/Train model on X Train Data
6. Evaluate Model on X Test Data
7. Adjust parameters as necessary and repeat steps 5 and 6
"""

X = df.drop("sales", axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

RMSE = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
print("Train|Test Ridge RMSE:", RMSE)


"""
Train | Validation | Test split
Perform train | test split twice to divide dataset in three parts
adjust the model on the validation set
Once satisfied evaluate metrics on test set
"""

X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=101)
X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=101)

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_eval = scaler.transform(X_eval)
X_test = scaler.transform(X_test)

model = Ridge(alpha=100)
model.fit(X_train, y_train)

y_eval_pred = model.predict(X_eval)

RMSE = root_mean_squared_error(y_eval, y_eval_pred)
print("Train|Validation|Test Ridge Validation:", RMSE)

model = Ridge(alpha=1)
model.fit(X_train, y_train)

y_eval_pred = model.predict(X_eval)

RMSE = root_mean_squared_error(y_eval, y_eval_pred)
print("Train|Validation|Test 2nd Ridge Validation :", RMSE)

y_pred = model.predict(X_test)
RMSE = root_mean_squared_error(y_test, y_pred)
print("Train|Validation|Test Ridge Fianl Eval :", RMSE)


"""
K-fold validation with cross_val_score
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=100)
scores = cross_val_score(model, X=X_train, y=y_train, scoring='neg_mean_squared_error', cv=5)
print("CrossValScore cv=5 alpha=100 RMSE:", np.sqrt(abs(scores.mean())))

model = Ridge(alpha=1)
scores = cross_val_score(model, X=X_train, y=y_train, scoring='neg_mean_squared_error', cv=5)
print("CrossValScore cv=5 alpha=1 RMSE:", np.sqrt(abs(scores.mean())))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
RMSE = root_mean_squared_error(y_test, y_pred)
print("CrossValScore cv=5 alpha=1 Ridge Fianl Eval :", RMSE)


"""
cross_validate function
"""

model = Ridge(alpha=100)
scores = cross_validate(model, X_train, y_train, scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error'],
                        cv=10)
scores = pd.DataFrame(scores)
# print("Cross Validate Scores:")
# print(scores)
# print("Cross validate mean values:")
# print(scores.mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
RMSE = root_mean_squared_error(y_test, y_pred)
print("CrossValidate cv=10 alpha=100 Ridge Fianl Eval :", RMSE)
