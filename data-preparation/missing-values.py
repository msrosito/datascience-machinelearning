################################################################################
# Missing values: removal and imputation
# Example: multiple linear regression
#
# See https://scikit-learn.org/stable/modules/impute.html
################################################################################

import datapreparation as dp
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

################################################################################
# Function definitions
################################################################################

"""
    generate_missing_values(df, n_missing):
Description
     Introduce missing data in a dataframe
Inputs
     `df`: dataframe (nobs x (n_features + 1))
     `n_missing` : number of missing values
     `missing_value`: missing value
Outputs
    `dff`: dataframe with missing values
"""

def generate_missing_values(df, n_missing):
    dfm = df.copy() 
    random.seed(0) # reproducible
    n_cols = len(dfm.columns)
    n_obs = len(dfm)
    for _ in range(n_missing):
        random_feature = random.randrange(n_cols)
        random_row = random.randrange(n_obs)
        dfm.iloc[random_row, random_feature] = missing_value
    return dfm
            
        
################################################################################
# Main
################################################################################

input_dir = '../data/'
filename = input_dir + 'data_example_MLR.csv'

# Read the data
df = pd.read_csv(filename)

# Generate dataframe with missing values
n = 80 # number of missing values
missing_value = -999999
dfm = generate_missing_values(df, n)

# Remove rows with missing values
df_remove = dfm[(dfm != missing_value).all(axis = 1)].reset_index()

# Impute values using the mean of each column
df_simple, mask = dp.simple_imputation_skl(dfm, 'mean', missing_value)

# Impute values using the mean of each column
k = 5 # number of nearest neighbors
df_knn, mask = dp.knn_imputation_skl(dfm, k, missing_value)

# Comparing multiple linear regressions

# Full dataset with no missing values
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
lr_full = LinearRegression().fit(X, y).score(X, y)

# Dataset removing rows with missing values
X = df_remove.iloc[:, :-1]
y = df_remove.iloc[:, -1]
lr_remove = LinearRegression().fit(X, y).score(X, y)

# Simple imputed dataset
X = df_simple.iloc[:, :-1]
y = df_simple.iloc[:, -1]
lr_simple = LinearRegression().fit(X, y).score(X, y)

# knn imputed dataset
X = df_knn.iloc[:, :-1]
y = df_knn.iloc[:, -1]
lr_knn = LinearRegression().fit(X, y).score(X, y)

# Print results
print('Multiple linear regression')
print('R2 using the dataset with no missing values ', lr_full)
print('R2 using the dataset removing rows with missing values ', lr_remove)
print('R2 using the simple imputed dataset (mean) ', lr_simple)
print('R2 using the 5nn imputed dataset ', lr_knn)
