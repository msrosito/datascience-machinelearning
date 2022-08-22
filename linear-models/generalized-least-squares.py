################################################################################
# Linear regression using generalized least squares
#
# See https://www.statsmodels.org/stable/examples/notebooks/generated/gls.html 
################################################################################

import numpy as np
import csv
import statsmodels.api as sm

################################################################################
# Function definitions
################################################################################


"""
    regression_generalized_least_squares_stm(X, y, cov, intercept = True):

Description
    Linear regression with correlated error terms using generalized least squares
Inputs
   `X: explatory variables (nobs x nfeat matrix).
   `y`: response variable (1D array).
   `intercept`: a constant should be added to the fitting (default: True)
Outputs
    `beta`: coefficients
    `beta0`: intercept
"""    
def regression_generalized_least_squares_stm(X, y, cov, intercept = True):
    if intercept:     
        X = sm.tools.tools.add_constant(X)        
        gls_model = sm.GLS(y, X, sigma = cov).fit()
        beta = gls_model.params
    else:
        gls_model = sm.GLS(y, X, sigma = cov).fit()
        beta = np.concatenate(([0.], gls_model.params))       
    print(gls_model.summary())
    return beta 


"""
    read_vars(filename)
   TODO: complete this
"""
def read_vars(filename):
    n = 50 # TODO: fix this
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
    
"""
    read_vars(filename)
   TODO: complete this
"""
def read_matrix(filename):
    n = 50 # TODO: fix this
    i = 0
    with open(filename) as data_file:
        csvreader = csv.reader(data_file)
        header = next(csvreader)
        cov = np.zeros((n, n))
        for row in csvreader:
            for j in range(n):
                cov[i, j] = row[j]
            i += 1
    
    return cov
    

################################################################################
# Main
################################################################################

input_dir = '../data/'
filename = input_dir + 'data_example_gls.csv'
cov_matrix_file = input_dir + 'cov_matrix_gls.csv'

print('Linear regression with correlated error terms')
# Read explatory and response variables
X, y = read_vars(filename)

# Read covariance matrix
cov_matrix = read_matrix(cov_matrix_file)
cond1 = np.all(np.linalg.eigvals(cov_matrix) > 0)
cond2 = np.all(cov_matrix - cov_matrix.T == 0)
# Check covariance matrix
if not cond1 or not cond2:
    print('The covariance matrix is not positive definite')

# Compute linear regression with generalized least squares and report results
beta = regression_generalized_least_squares_stm(X, y, cov_matrix, intercept = False)
print(beta)

# Using the formula

s = np.linalg.inv(cov_matrix)  
XXT = np.matmul(X, X.T)
beta = np.matmul(np.matmul(X.T, s), X)
beta = np.linalg.inv(beta)
beta = np.matmul(np.matmul(beta, X.T), s)
beta = np.matmul(beta, y)
print(beta)
