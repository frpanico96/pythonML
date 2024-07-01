import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

df = pd.read_csv('../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/penguins_size.csv')

print(df.head())
print(df['species'].unique())

"""Exploratory data analysis"""
print(df.isnull().sum())
print(df.info())
"""
Drop NaN columns since
they are relatively small percentage
relative to the whole dataset
"""
df = df.dropna()
print(df.info())
print(df.head())
print(df['island'].unique())
"""
Explore '.' value on 'sex' column
in order to assign a value 'female'/'male'
"""
print(df['sex'].unique())
print(df[df['sex'] == "."])
print(df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose())
df.at[336, 'sex'] = 'FEMALE'
print(df.loc[336])
"""
Plots
"""
# sns.pairplot(df, hue='species')
# sns.catplot(x='species', y='culmen_length_mm', data=df, kind='box', col='sex')
# plt.show()
"""
Convert 'island' column into dummy variables
"""
X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)
y = df['species']

"""
Start building decision tree
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
base_predictions = model.predict(X_test)

print(classification_report(y_test, base_predictions))
# matrix_plt = ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
# plt.show()

print(pd.DataFrame(index=X.columns, data=model.feature_importances_, columns=['Feature Importance'])
      .sort_values('Feature Importance'))

# plt.figure(figsize=(12,8))
# plot_tree(model, feature_names=X.columns, filled=True)
# plt.show()

def report_model(model, plot=False):
    model_preds = model.predict(X_test)
    print(classification_report(y_test, model_preds))
    print('\n')
    if plot:
        plt.figure(figsize=(12,8))
        plot_tree(model, feature_names=X.columns, filled=True)
        plt.show()


pruned_tree = DecisionTreeClassifier(max_depth=2)
pruned_tree.fit(X_train, y_train)
report_model(pruned_tree)

max_leaf_tree = DecisionTreeClassifier(max_leaf_nodes=3)
max_leaf_tree.fit(X_train, y_train)
report_model(max_leaf_tree)

entropy_tree = DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train, y_train)
report_model(entropy_tree, True)
