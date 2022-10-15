################################################################################
# Multiple linear regression using sklearn and statsmodels
#
# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
#     https://www.statsmodels.org/stable/regression.html
################################################################################


import sklearn.linear_model as lm 
import matplotlib.pyplot as plt
import numpy as np
import csv
import statsmodels.api as sm


################################################################################
# Function definitions
################################################################################

"""
    multiple_linear_regression_skl(x, y, intercept = True):

Description
    Classic multiple linear regression OLS using scikit learn
Inputs
   `X`: explanatory variables (nobs x nfeat matrix).
   `y`: response variable (1D array).
   `intercept`: a constant should be added to the fitting (default: True)
Outputs
    `beta`: coefficients
    `beta0`: intercept
    `R2`: coefficient of determination
"""
def multiple_linear_regression_skl(X, y, intercept = True):
    MLR = lm.LinearRegression(fit_intercept = intercept).fit(X, y)
    beta = MLR.coef_
    R2 = MLR.score(X, y)
    if intercept:
        beta0 = MLR.intercept_
    else:
        beta0 = 0.
    return beta0, beta, R2
    

"""
    multiple_linear_regression_stm(X, y, intercept = True):

Description
    Classic multiple linear regression OLS using statsmodels
Inputs
   `X: explanatory variables (nobs x nfeat matrix).
   `y`: response variable (1D array).
   `intercept`: a constant should be added to the fitting (default: True)
Outputs
    `beta`: coefficients
    `beta0`: intercept
    `R2`: coefficient of determination
"""    
def multiple_linear_regression_stm(X, y, intercept = True):
    if intercept:     
        X = sm.tools.tools.add_constant(X)        
        MLR = sm.OLS(y, X).fit()
        beta = MLR.params
        R2 = MLR.rsquared
    else:
        MLR = sm.OLS(y, X).fit()
        beta = MLR.params
        R2 = MLR.rsquared        
    print(MLR.summary())
    return beta[0], beta, R2


"""
    read_vars(filename)
   TODO: complete this
"""
def read_vars(filename):
    n = 100 # TODO: fix this
    y = np.zeros(n)
    i = 0
    with open(filename) as data_file:
        csvreader = csv.reader(data_file)
        header = next(csvreader)
        n_col = len(header)
        X = np.zeros((n, n_col - 1))
        for row in csvreader:
            for j in range(n_col - 1):
                X[i, j] = row[j]
            y[i] = row[n_col - 1]
            i += 1
    
    return X, y
    

################################################################################
# Main
################################################################################

input_dir = '../data/'
filename = input_dir + 'data_example_MLR.csv'

# Read explatory and response variables.
x, y = read_vars(filename)

# Example with scikit learn ####################################################
print('Multiple linear regression example (scikit learn)')

# Compute multiple linear regression
beta0, beta, R2 = multiple_linear_regression_skl(x, y)

# Report results
print('Intercept: ', beta0)
print('Coefficients: ', beta)
print('R2: ', R2)

# Example with statsmodel ######################################################
print('Multiple linear regression example (statsmodel)')

# Compute multiple linear regression and report results
beta0, beta, R2 = multiple_linear_regression_stm(x, y)

