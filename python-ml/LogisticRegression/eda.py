"""
Exploratory Data Analysis
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
from mpl_toolkits.mplot3d import Axes3D

age_feature = "age"
physical_score_feature = "physical_score"
test_result_label = "test_result"

df = pd.read_csv('../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/hearing_test.csv')

print(df.head())
print(df.describe())
print(df['test_result'].value_counts())

f1 = plt.figure(1)
sns.countplot(data=df, x="test_result")

f2 = plt.figure(2)
sns.boxplot(x=test_result_label, y=age_feature, data=df)

f3 = plt.figure(3)
sns.scatterplot(data=df, x=age_feature, y=physical_score_feature, hue=test_result_label)

f4 = plt.figure(4)
sns.pairplot(data=df, hue=test_result_label)

f5 = plt.figure(5)
ax = f5.add_subplot(111, projection='3d')

ax.scatter(df[age_feature], df[physical_score_feature], df[test_result_label])

plt.show()
