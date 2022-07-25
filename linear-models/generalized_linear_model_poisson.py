################################################################################
# GLM Poisson using sklearn
#
# See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor
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
    multiple_linear_model_poisson(x, y, intercept = True):

Description
    Generalized linear model with Poisson distribution using scikit learn
    ln(mu) = beta0 + beta1 * x, y ~ Poisson(mu)
Inputs
   `x`: explatory variable (nobs x nfeat matrix).
   `y`: response variable (1D array).
   `intercept`: a constant should be added to the fitting (default: True)
Outputs
    `beta`: coefficient
    `beta0`: intercept
"""
def generalized_linear_model_poisson(x, y, intercept = True):
    GLM = lm.PoissonRegressor(fit_intercept = intercept).fit(x, y)
    beta1 = GLM.coef_[0]
    if intercept:
        beta0 = GLM.intercept_
    else:
        beta0 = 0.
    return beta0, beta1
    
    
"""
    read_vars(filename) 1.373630605196558 0.7630392138264279
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
    plot_poisson_linear_regression(x, y, intercept = True):

Description
    Scatter plot of the data + regression line of expected values of y
Inputs
   `x`: explatory variables
   `y`: response variable
   `beta0`: intercept
   `beta1`: slope
   `output_dir`: output directory
"""
def plot_poisson_linear_regression(x, y, beta0, beta1, output_dir):
    liminfx = min(x)
    limsupx = max(x)
    liminfy = np.exp(beta1 * liminfx + beta0)
    limsupy = np.exp(beta1 * limsupx + beta0)
    xax = np.linspace(liminfx, limsupx, 20)
    yax = np.exp(beta1 * xax + beta0)
    fs = 16
    plt.plot(x, y, marker = 'o', linewidth = 0, c = 'm')
    plt.plot(xax, yax, c = 'b', linewidth = 2)
    plt.xticks(fontsize = 0.75 * fs)
    plt.yticks(fontsize = 0.75 * fs)
    plt.title('Results ', fontsize = fs)
    plt.xlabel('x', fontsize = fs)
    plt.ylabel('y', fontsize = fs)
    plt.text(liminfx, limsupy * 1.25, 'y ~ Poisson(exp(' +str(round(beta0, 3)) + '+' + str(round(beta1, 3)) + ' * x))', 
    fontsize = 0.75 * fs)
    plt.savefig(output_dir + 'generalized_linear_regression_poisson.png')
    plt.close()
    

################################################################################
# Main
################################################################################

input_dir = 'data/'
output_dir = 'results_LR/'
filename = input_dir + 'data_example_GLMP.csv'

# Read explatory and response variables.
x, y = read_vars(filename)

print('Generalized linear model with Poisson distribution example')

# Compute simple linear regression
beta0, beta1 = generalized_linear_model_poisson(x, y)

# Report results
print('Intercept: ', beta0)
print('Coefficient: ', beta1)

# Plot results
plot_poisson_linear_regression(x, y, beta0, beta1, output_dir)
