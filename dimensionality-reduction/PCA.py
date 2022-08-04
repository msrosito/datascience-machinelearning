################################################################################
# Example of Principal Component Analysis
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit
################################################################################

import sklearn.decomposition as dsl
import numpy as np
import csv
import matplotlib.pyplot as plt

################################################################################
# Function definitions
################################################################################

"""
    principal_components_skl(X):

Description
    Analysis of principal components of a matrix of features using scikit-learn
Input
   `X`: matrix of features (nobs x nfeatures). X must be centered around its mean
Outputs
    `components`: axis in the feature space
    `exp_variance`: percentage of variance explained by each component
    `T`: projection of each observation in the principal components (nobs x nfeatures)
"""   

def principal_components_skl(X):
    pca = dsl.PCA().fit(X)
    components = pca.components_
    ex_variance = pca.explained_variance_ratio_
    T = pca.transform(X)
    return components, ex_variance, T
    
    
################################################################################
# Main
################################################################################    
    
output_dir = 'results_dimensionality_reduction/'
input_dir = '../data/' 

# Read the data to a centered numpy array
filename = input_dir + 'bivariate_normal.csv'
f = open(filename, 'rb')
X = np.loadtxt(f, delimiter = ',')
X = np.delete(X, (0), axis = 0)
X = X - X.mean(axis = 0)

# About the input data
print('Correlated bivariate gaussian variables') 

# Compute the principal components
c, ev, T = principal_components_skl(X)

# Report results

print('Principal component 1 (PC1): ', c[0])
print('Porcentage of the variance explained by PC1: ', np.round(ev[0], 3) * 100, '%')
print('Principal component 2: (PC2): ', c[1])
print('Porcentage of the variance explained by PC2: ', np.round(ev[1], 3) * 100, '%')

# Plot

m1 = c[0][1] / c[0][0]
m2 = c[1][1] / c[1][0]

ex = np.array([-5., -2., 1., 0., 1., -2., 5.])

plt.plot(X.T[0], X.T[1], 'o', linewidth = 0)
plt.plot(ex, m1 * ex, linewidth = 2)
plt.plot(ex, m2 * ex, linewidth = 2)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.savefig(output_dir + 'PCA_bivariate_normal.png')
