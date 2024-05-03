"""
 outliers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_ages(mu=50, sigma=13, num_samples=100, seed=42):

    np.random.seed(seed)

    sample_ages = np.random.normal(loc=mu, scale=sigma, size=num_samples)
    sample_ages = np.round(sample_ages, decimals=0)

    return sample_ages



sample = create_ages()

# sns.displot(sample)
# sns.boxplot(sample)
#
# plt.show()

ser = pd.Series(sample)

# Brute force method to remove outlier
print(ser.describe())
IQR = 55.25 - 42.00
lower_limit = 42.0 - 1.5*IQR
print(lower_limit)

ser_without_outlier = ser[ser > lower_limit]
print(ser_without_outlier)


# Remove outliers with np.percentile()
q75, q25 = np.percentile(sample, [75, 25])
IQRnp = q75 - q25
lower_limit_np = q25 - 1.5*IQRnp
print(lower_limit_np)
ser_without_outlier_np = ser[ser > lower_limit_np]
print(ser_without_outlier_np)