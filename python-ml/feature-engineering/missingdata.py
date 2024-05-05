"""
Missing Data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/Ames_Housing_Feature_Description.txt") as f:
    print(f.read())

df = pd.read_csv("../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/Ames_outliers_removed.csv")
print(df.info())

"""
Unique values such as IDs can be used to index the DF
Or we can just drop them
"""

df = df.drop("PID", axis=1)

"""
Explore null values
"""


# print(df.isnull().sum())

def percent_missing(dataframe):
    percent_nan = 100 * dataframe.isnull().sum() / len(dataframe)
    percent_nan = percent_nan[percent_nan > 0].sort_values()

    return percent_nan

percent_nan = percent_missing(dataframe=df)
print(percent_nan)

print(percent_nan[percent_nan < 1])
# print(df[df["Electrical"].isnull()])
# print(df[df["Bsmt Half Bath"].isnull()])
"""
Drop rows where the subset features are missing less than 1% in the entire df
"""
df = df.dropna(axis=0, subset=["Electrical", "Garage Cars"])

"""
Basement missing features are due to the fact that the basement is missing
Instead of dropping the feature is possible to fill them with 0
"""
percent_nan = percent_missing(dataframe=df)
print(percent_nan[percent_nan < 1])


# Basement Numeric columns --> fillna with 0
bsmt_num_cols = ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "Bsmt Half Bath", "Bsmt Full Bath"]

df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

# Basement String columns
bsmt_str_cols = ["Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"]
df[bsmt_str_cols] = df[bsmt_str_cols].fillna("None")


"""
Masonry missing data is due to missing masonry
It makes sense to fill them with "None" 
"""
msnry_num_cols = ["Mas Vnr Area"]
df[msnry_num_cols] = df[msnry_num_cols].fillna(0)

msnry_str_cols = ["Mas Vnr Type"]
df[msnry_str_cols] = df[msnry_str_cols].fillna("None")

"""
Recalculate percentage of nan after cleaning missing data
with low percetange of missing
"""
percent_nan = percent_missing(dataframe=df)
sns.barplot(x=percent_nan.index, y=percent_nan)
plt.xticks(rotation=90)
# plt.ylim(0, 1)
plt.xlabel("Feature")
plt.ylabel("Missing Percentage")
plt.tight_layout()
plt.show()


