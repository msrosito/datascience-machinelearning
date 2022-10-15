################################################################################
# Linear regression adding L1 and L2 regularization using scikit-learn

# See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
#      https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
################################################################################

import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
import pandas as pd

################################################################################
# Function definitions
################################################################################

"""
    linear_regression_skl(X_train, y_train, X_test, y_test)

Description
    Classic linear regression OLS using scikit learn
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels (ntest array)
Outputs
    `coefficients`: coefficients (nfeatures)
    `intercept`: intercept
    `R2_train`: coefficient of determination for the training sample
    `R2_test`: coefficient of determination for the test sample    
"""

def linear_regression_skl(X_train, y_train, X_test, y_test):
    regression = lm.LinearRegression().fit(X_train, y_train)
    coefficients = regression.coef_
    intercept = regression.intercept_
    R2_train = regression.score(X_train, y_train)
    R2_test = regression.score(X_test, y_test)
    return coefficients, intercept, R2_train, R2_test
    
    
"""
    linear_regression_lasso_skl(X_train, y_train, X_test, y_test)

Description
    Linear regression with L1 regulatization
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels (ntest array)
   `alpha`: constant than multiplies the L1 term in the loss function
Outputs
    `coefficients`: coefficients (nfeatures)
    `intercept`: intercept
    `R2_train`: coefficient of determination for the training sample
    `R2_test`: coefficient of determination for the test sample    
"""

def linear_regression_lasso_skl(X_train, y_train, X_test, y_test, alpha):
    regression = lm.Lasso(alpha, random_state = 0).fit(X_train, y_train)
    coefficients = regression.coef_
    intercept = regression.intercept_
    R2_train = regression.score(X_train, y_train)
    R2_test = regression.score(X_test, y_test)        
    return coefficients, intercept, R2_train, R2_test
    

"""
    linear_regression_ridge_skl(X_train, y_train, X_test, y_test)

Description
    Linear regression with L2 regulatization
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels (ntest array)
   `alpha`: constant than multiplies the L2 term in the loss function
Outputs
    `coefficients`: coefficients
    `intercept`: intercept
    `R2_train`: coefficient of determination for the training sample
    `R2_test`: coefficient of determination for the test sample    
"""    
    
def linear_regression_ridge_skl(X_train, y_train, X_test, y_test, alpha):
    regression = lm.Ridge(alpha, random_state = 0).fit(X_train, y_train)
    coefficients = regression.coef_
    intercept = regression.intercept_
    R2_train = regression.score(X_train, y_train)
    R2_test = regression.score(X_test, y_test)        
    return coefficients, intercept, R2_train, R2_test      
        
################################################################################
# Function definitions
################################################################################

input_dir = '../data/'

# Read the data and construct the training and test datasets
data = pd.read_csv(input_dir + 'data_example_reg.csv', names = ['feat1', 'feat2', 'feat3', 'res'])
data['feat4'] = data['feat1'] * data['feat2'] * data['feat3'] 
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Example of classical linear regression
print('Classical linear regression')
# Compute the linear regressions and the scores
coef, intercept, R2_train, R2_test = linear_regression_skl(X_train, y_train, X_test, y_test)
# Report results
print('Coefficients and intercept: ', coef, intercept)
print('Coefficient of determination (train): ', R2_train)
print('Coefficient of determination (test): ', R2_test)
 
# L1 and L2 regularization
alpha = 0.1
# Example of linear regression using L1 regulatization
print('Linear regression with L1 regularization')
# Compute the linear regressions and the scores
coef, intercept, R2_train, R2_test = linear_regression_lasso_skl(X_train, y_train, X_test, y_test, alpha)
# Report results
print('Coefficients and intercept: ', coef, intercept)
print('Coefficient of determination (train): ', R2_train)
print('Coefficient of determination (test): ',  R2_test)
# Example of linear regression using L2 regulatization
print('Linear regression with L2 regularization')
# Compute the linear regressions and the scores
coef, intercept, R2_train, R2_test = linear_regression_ridge_skl(X_train, y_train, X_test, y_test, alpha)
# Report results
print('Coefficients and intercept: ', coef, intercept)
print('Coefficient of determination (train): ', R2_train)
print('Coefficient of determination (test): ' , R2_test)
