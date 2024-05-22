"""
KNN Model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline

df = pd.read_csv('../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/gene_expression.csv')

print(df.head())

# sns.scatterplot(data=df, x='Gene One', y='Gene Two', hue='Cancer Present', alpha=0.5)
# plt.show()

X = df.drop('Cancer Present', axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
Model Training
"""

model = KNeighborsClassifier(n_neighbors=1)

model.fit(X_train, y_train)

"""
Evaluate model
"""

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print(class_report)

"""Elbow method"""

test_error_rates = []

for k in range(1, 30):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    test_error_rates.append(error)

# plt.plot([k for k in range(1, 30)], test_error_rates, label="Elbow method")
# plt.axhline(y=0.066, color="orange", linestyle="-.")
# plt.axhline(y=0.058, color="orange", linestyle="-.")
# plt.axhline(y=0.0598, color="red", linestyle="-.", label="Best K=6")
# plt.xlabel("K")
# plt.ylabel("Error rate")
# plt.xlim(0.9, 29.1)
# plt.legend()
# plt.show()

"""CrossValidation method"""

# Pipeline for GridCV

scaler = StandardScaler()
model = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

operations = [
    ('scaler', scaler),
    ('model', model),
]

pipe = Pipeline(steps=operations)

k_values = list(range(1, 20))
# In order to work with pipeline
# The syntax should be
# variable-associated-with-model__parameter-name
# ex. model__n_neighbors
param_grid = {
    'model__n_neighbors': k_values
}

full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
full_cv_classifier.fit(X_train, y_train)
# print(full_cv_classifier.best_estimator_.get_params())
full_pred = full_cv_classifier.predict(X_test)

print(classification_report(y_test, full_pred))

new_patient = [[3.8, 6.4]]
new_patient_pred = full_cv_classifier.predict(new_patient)
new_patient_prob = full_cv_classifier.predict_proba(new_patient)
print(new_patient_pred, new_patient_prob)