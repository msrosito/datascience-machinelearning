################################################################################
# Multiple linear regression using sklearn and statsmodels
#
# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
################################################################################

from sklearn.linear_model import LinearRegression
import sklearn.tree as tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import csv


################################################################################
# Function definitions
################################################################################

"""
    multiple_linear_regression_scores_skl(X_train, y_train, X_test, y_test):

Description
    Classic multiple linear regression OLS using scikit learn
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels (ntest x nfeatures matrix)
Outputs
    `R2_train`: coefficient of determination for the training sample
    `R2_test`: coefficient of determination for the test sample    
"""
def multiple_linear_regression_skl(X_train, y_train, X_test, y_test):
    MLR = LinearRegression().fit(X, y)
    return MLR.score(X_train, y_train), MLR.score(X_test, y_test)


"""
    decision_tree_regressor_scores_skl(X_train, y_train, X_test, y_test):

Description
    Decision trees for regression. Criterion: squared error.
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix), optional
   `y_test`: test set of true labels (ntest x nfeatures matrix), optional
Outputs
    `decision_tree`: decision tree
    `R2_train`: coefficient of determination for the training sample
    `R2_test`: coefficient of determination for the test sample   
     
""" 
def decision_tree_regression_skl(X_train, y_train, X_test = None,
    y_test = None):
    decision_tree = tree.DecisionTreeRegressor(criterion = 'mse', max_depth = 5,
    random_state = 0).fit(X_train, y_train)
    y_pred_train = decision_tree.predict(X_train)
    return decision_tree, y_pred_train, decision_tree.score(X_train, y_train), decision_tree.score(X_test, y_test)

        
'''
    plot_decision_tree(df, filename, title, feature_names = None, 
    class_names = None)

Description:
    Plot of a decision tree
Inputs
    `df`: decision tree estimator
    `filename`: filename of the plot
    `title`: title for the plot
    `features_names`: names of the features, optional
    `class_names`: labels for the target variable, optional
'''
def plot_decision_tree(df, filename, title):
    f = plt.figure()
    f.set_figwidth(30)
    f.set_figheight(30)
    tree.plot_tree(dt, filled = True) 
    plt.title(title)
    plt.savefig(filename) 
    
    
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
output_dir = 'resultsLM/'
filename = input_dir + 'data_example_MLR.csv'

# Read explatory and response variables.
X, y = read_vars(filename)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 0)

# Compute multiple linear regression
R2_train_lr, R2_test_lr = multiple_linear_regression_skl(X_train, y_train, X_test, y_test)

# Compute dt regression
print('DT example (scikit learn)')
dt, y_pred, R2_train_dt, R2_test_dt = decision_tree_regression_skl(X_train, y_train, X_test, y_test)

# Report results
print('Multiple linear regression example (scikit learn)')
print('Coefficient of determination (train): ', R2_train_lr)
print('Coefficient of determination (test): ', R2_test_lr)
print('Decision tree regression example (scikit learn)')
print('Coefficient of determination (train): ', R2_train_dt)
print('Coefficient of determination (test): ', R2_test_dt)

# Plot
filename = output_dir + 'regression-tree-example.png'
title = 'Regression Tree'
plot_decision_tree(dt, filename, title)
