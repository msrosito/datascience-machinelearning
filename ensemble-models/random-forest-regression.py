################################################################################
# Regression using Random Forest
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
################################################################################

from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as skm
import numpy as np
import csv 

################################################################################
# Function definitions
################################################################################

"""
    random_forest_skl(n_estimators, max_samples, X_train, y_train, X_test = None, 
    y_test = None)

Description
    Random Forest for regression using boostrap. Criterion: squared error
Inputs
   `nt`: number of trees
   `max_samples`: number of samples of X_train to train each tree
   `min_samples_split`: minimum number of samples required to split an internal node 
   Default: 3
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix), optional
   `y_test`: test set of true labels (ntest x nfeatures matrix), optional
Outputs
    `f_importance`: impurity based features importance
    `y_train_pred`: predicted values for the training sample (ntrain array)
    `R2_train`: coefficient of determination for the training sample
    `y_test_pred`: predicted values for the test sample (ntrain array)
    `R2_train`: coefficient of determination for the test sample

""" 
def random_forest_regression_skl(nt, max_samples, X_train, y_train, 
    X_test, y_test, min_samples_split = 3):
    random_forest = RandomForestRegressor(n_estimators = nt, max_samples = max_samples,
    min_samples_split = min_samples_split, random_state = 0).fit(X_train, y_train) # reproducible
    f_importance = random_forest.feature_importances_    
    y_train_pred = random_forest.predict(X_train)
    R2_train = random_forest.score(X_train, y_train)
    y_test_pred = random_forest.predict(X_test)
    R2_test = random_forest.score(X_test, y_test) 
    return f_importance, y_train_pred, R2_train, y_test_pred, R2_test


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
X, y = read_vars(filename)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 0)

# Compute the regression using random forest
nt = 10
ms = 20
f_importance, y_train_pred, R2_train, y_test_pred, R2_test = random_forest_regression_skl(nt, ms, X_train, 
y_train, X_test, y_test)

# Compute multiple linear regression for comparison
MLR = LinearRegression().fit(X_train, y_train)
R2_train_lr = MLR.score(X_train, y_train)
R2_test_lr = MLR.score(X_test, y_test)

# Report results
print('Regression using random forest')
print('Feature importance: ')
for i in range(len(f_importance)):
    print('Feature ', i, 'importance :', round(f_importance[i], 3) * 100, '%')
print('Coefficient of determination (train): ', R2_train)
print('Coefficient of determination (test): ', R2_test)
print('Linear regression for comparison')
print('Coefficient of determination (train): ', R2_train_lr)
print('Coefficient of determination (test): ', R2_test_lr)
