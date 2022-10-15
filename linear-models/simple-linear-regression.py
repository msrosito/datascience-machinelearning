################################################################################
# Simple linear regression using sklearn and statsmodels
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
    simple_linear_regression_skl(x, y, intercept = True):

Description
    Classic simple linear regression OLS using scikit learn
Inputs
   `x`: explanatory variables (nobs x 1 matrix).
   `y`: response variable (1D array).
   `intercept`: a constant should be added to the fitting (default: True)
Outputs
    `beta1`: slope
    `beta0`: intercept
    `R2`: coefficient of determination
"""
def simple_linear_regression_skl(x, y, intercept = True):
    SLR = lm.LinearRegression(fit_intercept = intercept).fit(x, y)
    beta1 = SLR.coef_[0]
    R2 = SLR.score(x, y)
    if intercept:
        beta0 = SLR.intercept_
    else:
        beta0 = 0.
    return beta0, beta1, R2
    

"""
    simple_linear_regression_stm(x, y, intercept = True):

Description
    Classic simple linear regression OLS using statsmodels
Inputs
   `x`: explanatory variables (nobs x 1 matrix).
   `y`: response variable (1D array).
   `intercept`: a constant should be added to the fitting (default: True)
Outputs
    `beta1`: slope
    `beta0`: intercept
    `R2`: coefficient of determination
"""    
def simple_linear_regression_stm(x, y, intercept = True):
    if intercept:     
        x = sm.tools.tools.add_constant(x)        
        SLR = sm.OLS(y, x).fit()
        beta = SLR.params
        R2 = SLR.rsquared
    else:
        SLR = sm.OLS(y, x).fit()
        beta = [0., SLR.params[0]]
        R2 = SLR.rsquared        
    print(SLR.summary())
    return beta[0], beta[1], R2


"""
    read_vars(filename)
   TODO: complete this
"""
def read_vars(filename):
    n = 100 # TODO: fix this
    x = np.zeros(n)
    y = np.zeros(n)
    i = 0
    with open(filename) as data_file:
        csvreader = csv.reader(data_file)
        next(csvreader)
        for row in csvreader:
            x[i] = row[0]
            y[i] = row[1]
            i += 1
    
    x = x.reshape(-1, 1)
#    y = y.reshape(-1,1)    
    
    return x, y


"""
    plot_simple_linear_regression(x, y):

Description
    Scatter plot of the data + regression line
Inputs
   `x`: explanatory variable
   `y`: response variable
   `beta0`: intercept
   `beta1`: slope
   `label`: library used to compute the linear regression
   `output_dir`: output directory
"""
def plot_simple_linear_regression(x, y, beta0, beta1, label, output_dir):
    liminfx = min(x)
    limsupx = max(x)
    liminfy = beta1 * liminfx + beta0
    limsupy = beta1 * limsupx + beta0
    xax = np.linspace(liminfx, limsupx, 5)
    yax = beta1 * xax + beta0
    fs = 16
    plt.plot(x, y, marker = 'o', linewidth = 0, c = 'm')
    plt.plot(xax, yax, c = 'b', linewidth = 2)
    plt.xticks(fontsize = 0.75 * fs)
    plt.yticks(fontsize = 0.75 * fs)
    plt.title('Results ' + label, fontsize = fs)
    plt.xlabel('x', fontsize = fs)
    plt.ylabel('y', fontsize = fs)
    plt.text(liminfx, limsupy * 1.1, str(round(beta0, 3)) + '+' + str(round(beta1, 3)) + ' * x ', 
    fontsize = 0.75 * fs)
    plt.savefig(output_dir + 'simple_linear_regression_' + label + '.png')
    plt.close()
    

################################################################################
# Main
################################################################################

input_dir = '../data/'
output_dir = 'resultsLM/'
filename = input_dir + 'data_example_LR.csv'

# Read explatory and response variables.
x, y = read_vars(filename)

# Example with scikit learn ####################################################
print('Simple linear regression example (scikit learn)')

# Compute simple linear regression
beta0, beta1, R2 = simple_linear_regression_skl(x, y)

# Report results
print('Intercept: ', beta0)
print('Coefficient: ', beta1)
print('R2: ', R2)

# Plot results
label = "scikit"
plot_simple_linear_regression(x, y, beta0, beta1, label, output_dir)

# Example with statsmodel ######################################################
print('Simple linear regression example (statsmodel)')

# Compute simple linear regression and report results
beta0, beta1, R2 = simple_linear_regression_stm(x, y)

# Plot results
label = "statsmodel"
plot_simple_linear_regression(x, y, beta0, beta1, label, output_dir)
