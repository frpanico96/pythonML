"""
Logistic Regression

with multi Multi-Class problems

Working with Iris Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve,
                             auc, RocCurveDisplay)


df = pd.read_csv("../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/iris.csv")

# print(df.head())

"""
Exploratory Data Analysis
"""

print(df.describe())
print(df['species'].value_counts())

# Data Visualization plots
#sns.countplot(data=df, x="species")
#sns.scatterplot(x="petal_length", y="petal_width", data=df, hue="species")
#sns.pairplot(data=df, hue="species")
#sns.heatmap(df.corr(numeric_only=True), annot=True)
#plt.show()


# Model
# Train with GridSearch process

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000)

# build penalty grid
penalty = ['l1', 'l2', 'elasticnet']
l1_ratio = np.linspace(0, 1, 20)
C = np.logspace(0, 10, 20)

param_grid = {'penalty': penalty,
              'l1_ratio': l1_ratio,
              'C': C}

grid_model = GridSearchCV(estimator=log_model, param_grid=param_grid)

grid_model.fit(X_train, y_train)

# Evaluate metrics
print("Best Param:", grid_model.best_params_)

y_pred = grid_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# conf_matrix = ConfusionMatrixDisplay.from_estimator(estimator=grid_model, X=X_test, y=y_test)
# plt.show()

def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(5,5)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


plot_multiclass_roc(clf=grid_model, X_test=X_test, y_test=y_test, n_classes=3)