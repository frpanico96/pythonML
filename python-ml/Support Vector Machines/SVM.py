"""
Support Vector Machines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Function to plot margins
def plot_svm_boundary(model, X, y):
    X = X.values
    y = y.values

    # Scatter Plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='seismic')

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


df = pd.read_csv("../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/mouse_viral_study.csv")
print(df.head())

# sns.scatterplot(x='Med_1_mL', y="Med_2_mL", hue="Virus Present", data=df)
# # hyperplane
# x = np.linspace(0, 10, 100)
# m = -1
# b = 11
# y = m*x + b
# plt.plot(x, y, 'black')
# plt.show()

X = df.drop("Virus Present", axis=1)
y = df["Virus Present"]

# linear kernel
# model = SVC(kernel='linear', C=1000)

# sigmoid kernel
# model = SVC(kernel='sigmoid')

# polynomial kernel
# model = SVC(kernel='poly', C=0.05, degree=4)

# rbf kernel (default)
# model = SVC(kernel='rbf', C=1, gamma='auto')
# model.fit(X, y)
# plot_svm_boundary(model, X, y)

# Grid Search
# svm = SVC()
# param_grid = {
#    'C': [0.01, 0.1, 1],
#    'kernel': ['linear', 'rbf']
# }

# grid = GridSearchCV(svm, param_grid)
# grid.fit(X, y)

# print(grid.best_params_)


"""
Regression tasks
"""
df = pd.read_csv("../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/cement_slump.csv")
print(df.head())

label = "Compressive Strength (28-day)(Mpa)"
# plt.figure(figsize=(8, 8))
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

X = df.drop(label, axis=1)
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

base_model = SVR()

base_model.fit(X_train, y_train)
base_preds = base_model.predict(X_test)
print("Base Best Params:", base_model.get_params())
print("Base MAE:", mean_absolute_error(y_test, base_preds))
print("Base RMSE:", np.sqrt(mean_squared_error(y_test, base_preds)))

param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'epsilon': [0, 0.01, 0.1, 0.5, 1, 2],
}

svr = SVR()
grid = GridSearchCV(svr, param_grid)
grid.fit(X_train, y_train)


grid_preds = grid.predict(X_test)
print("Best Params:", grid.best_params_)
print("MAE:", mean_absolute_error(y_test, grid_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, grid_preds)))

sns.clustermap()
