"""
Creating and training the model
The dataset in exam has two features
* Age
* Physical_score

And a label
* test_result
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay,
                             precision_score,
                             recall_score,
                             RocCurveDisplay,
                             PrecisionRecallDisplay)

age_feature = "age"
physical_score_feature = "physical_score"
test_result_label = "test_result"

df = pd.read_csv('../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/hearing_test.csv')
"""
Separate X and y
create test|train split
scale data fitting only on X_train
"""
X = df.drop(test_result_label, axis=1)
y = df[test_result_label]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
train model
"""

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

print("Coeff:", log_model.coef_)

"""
predict
"""
y_pred = log_model.predict(X_test)
# print("Predictions:", y_pred)

"""
Performance evaluation
"""

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
matrix_plt = ConfusionMatrixDisplay.from_estimator(estimator=log_model, X=X_test, y=y_test, normalize="all")
# matrix_plt.plot()
# plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report")
print(class_report)

# Precision
prec = precision_score(y_test, y_pred)
print("Precision:", prec)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# ROC curve
roc = RocCurveDisplay.from_estimator(estimator=log_model, X=X_test, y=y_test)
#roc.plot()
#plt.show()

# Precision Recall Curve
prec_recall = PrecisionRecallDisplay.from_estimator(estimator=log_model, X=X_test, y=y_test)
#prec_recall.plot()
#plt.show()

# Probabilities

prob = log_model.predict_proba(X_test)
print(prob[0])