"""
Missing Data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def percent_missing(dataframe):
    percent_nan = 100 * dataframe.isnull().sum() / len(dataframe)
    percent_nan = percent_nan[percent_nan > 0].sort_values()

    return percent_nan


bsmt_num_cols = ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "Bsmt Half Bath", "Bsmt Full Bath"]
bsmt_str_cols = ["Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"]
msnry_num_cols = ["Mas Vnr Area"]
msnry_str_cols = ["Mas Vnr Type"]
gar_str_cols = ["Garage Type", "Garage Finish", "Garage Qual", "Garage Cond"]
gar_num_cols = ["Garage Yr Blt"]
high_perc_nan_cols = ["Fence", "Pool QC", "Misc Feature", "Alley"]
fireplace_cols = ["Fireplace Qu"]

lot_frontage_col = "Lot Frontage"
neighborhood_col = "Neighborhood"
ms_subclass_col = "MS SubClass"

perc_nan_x_label = "Feature"
perc_nan_y_label = "Percentage Nan"

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
# print(percent_nan[percent_nan < 1])


# Basement Numeric columns --> fillna with 0

df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

# Basement String columns
df[bsmt_str_cols] = df[bsmt_str_cols].fillna("None")

"""
Masonry missing data is due to missing masonry
It makes sense to fill them with "None" 
"""
df[msnry_num_cols] = df[msnry_num_cols].fillna(0)

df[msnry_str_cols] = df[msnry_str_cols].fillna("None")

"""
Garage Feature --> Fill values
"""

df[gar_str_cols] = df[gar_str_cols].fillna("None")

df[gar_num_cols] = df[gar_num_cols].fillna(0)

"""
Fence, Alley, Misc Feature, Pool QC
Those features are missing in more than 80% of data
In this case, it is reasonable to drop the features
"""

df = df.drop(high_perc_nan_cols, axis=1)

"""
Fireplace QU feature is in a percentage zone
Where it does not make sense to drop the feature nor removing rows
From the feature description we can infer that not having the value
means not having the feature, so we can fill the non values
"""
# print(df.head())
df[fireplace_cols] = df[fireplace_cols].fillna("None")

"""
Lot Frontage feature is in a percentage zone
Where it does not make sense to drop the feature nor removing rows
We can assume that LotFrontage depends on the Neighborhood
Group the df by Neighborhood and extract mean value of Lot Frontage
We can then fill missing value with mean value
Fill residuals value with 0 (lot frontage in those residuals cases is meaningless)
"""

df[lot_frontage_col] = df.groupby(neighborhood_col)[lot_frontage_col].transform(
    lambda value: value.fillna(value.mean()))
df[lot_frontage_col] = df[lot_frontage_col].fillna(0)
# sns.boxplot(x="Lot Frontage", y="Neighborhood", data=df, orient='h')
# plt.tight_layout()
# plt.show()

print(df.isnull().sum())

"""
Recalculate percentage of nan after cleaning missing data
with low percetange of missing
"""
# percent_nan = percent_missing(dataframe=df)
# sns.barplot(x=percent_nan.index, y=percent_nan)
# plt.xticks(rotation=90)
# # plt.ylim(0, 1)
# plt.xlabel(perc_nan_x_label)
# plt.ylabel(perc_nan_y_label)
# plt.tight_layout()
# plt.show()

"""
Dealing with Categorical data

MSSubclass feature as a numerical encoding
But it does not have a clear relationship
And can bring to create implied ordering and relationships
We want to transform it in a string encoded feature
"""

df = pd.read_csv("../../UNZIP_FOR_NOTEBOOKS_FINAL/DATA/Ames_NO_Missing_Data.csv")

df[ms_subclass_col] = df[ms_subclass_col].apply(str)
# Pandas treat string as objects
# print(df.select_dtypes(include='object'))
object_df = df.select_dtypes(include='object')
numeric_df = df.select_dtypes(exclude='object')
df_object_dummies = pd.get_dummies(object_df, drop_first=True)

df = pd.concat([numeric_df, df_object_dummies], axis=1)
print(df.corr(numeric_only=True)["SalePrice"].sort_values())