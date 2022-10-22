################################################################################
# Functions for data preparation: missing values, preprocessing, outliers
#
# See: https://scikit-learn.org/stable/modules/preprocessing.html
#      https://scikit-learn.org/stable/modules/impute.html
#      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html
################################################################################

import sklearn.preprocessing as pp
import sklearn.impute as imp
from scipy.stats.mstats import winsorize
import pandas as pd

################################################################################
# Function definitions
################################################################################

"""
    simple_imputatior_skl(X, strategy)
Description
     Missing values imputation using mean, median, most_frequent, constant
Inputs
     `data`: dataframe with missing values (nobs x (n_features + 1))
     `strategy` : imputation strategy. Use 'mean', 'median' or 'constant' for continous variables
     and 'most_frequent' or 'constant' for categorical variables (nobs x (n_features + 1)).
     `missing_value`: missing value
Outputs
    `dataf`: dataframe with imputed values
    `mask`: boolean matrix indicating missing values
"""
 
def simple_imputatior_skl(data, strategy, missing_value):
    mask = imp.MissingIndicator(missing_values = missing_value).fit_transform(data)
    dataf = imp.SimpleImputer(missing_values = missing_value, strategy = strategy).fit_transform(data)
    dataf = pd.DataFrame(dataf)
    return dataf, mask

    
"""
    knn_imputation_skl(X, k, w = 'uniform')
Description
     Missing values imputation using k-nearest neighbors
Inputs
     `data`: dataframe with missing values (nobs x (n_features + 1))
     `k` : number of neighbors
     `missing_value`: missing value
     `w`: weight for each knn ('uniform', 'distance', 'callable'). Default = 'uniform'.
Outputs
    `dataf`: dataframe with imputed values
    `mask`: boolean matrix indicating missing values
"""

def knn_imputation_skl(data, k, missing_value, w = 'uniform'):
    mask = imp.MissingIndicator(missing_values = missing_value).fit_transform(data)
    dataf = imp.KNNImputer(n_neighbors = k, missing_values = missing_value, weights = w).fit_transform(data)
    dataf = pd.DataFrame(dataf)
    return dataf, mask
    

"""
    data_preprocessing(X)
Description
    Data preprocessing
Input
    `X`: data features (nsamples x nfeatures)
Outputs
    `X_center`: X transformed by centering each feature to the mean data features 
    `X_stand`: X transformed by centering each feature to the mean and scaling to unit variance
    `X_norm:` : X transformed so that each sample has l2 norm equal to 1
    `X_minmax`: X transformed by scaling each feature to the range (0, 1)
    `X_minabs`: X transformed by scaling each feature to the range [-1, 1]
"""
    
def data_preprocessing(X):
    X_center = pp.scale(X, with_std = False) 
    X_stand = pp.scale(X)
    X_norm = pp.normalize(X, norm = 'l2')
    X_minmax = pp.minmax_scale(X, feature_range = (0, 1))
    X_maxabs = pp.maxabs_scale(X)
    return X_center, X_stand, X_norm, X_minmax, X_maxabs
    
    
"""    
    data_preprocessing_with_outliers(X, quantile_inf = 0.25, quantile_sup = 0.75)
Description
    Useful preprocessing for data with outliers
Inputs
    `X`: data features (nsamples x nfeatures)
    `quantile_inf`: lower limit for winsorization and lower quantile range limit for robust scale
    Default 0.25
    `quantile_sup`: upper limit for winsorization and upper quantile range limit for robust scale
    Default 0.75
Outputs
    `X_winsorize`: winsorized version of X setting the lowest values to quantile_inf 
    and the highest to quantile_sup
    `X_rob_scale`:  X transformed by centering each feature to the median and scaling to the IQR
    IQR = quantile_sup - quantile_inf (both median and IQR are robust to outliers)   
"""    
def data_preprocessing_with_outliers(X, quantile_inf = 0.25, quantile_sup = 0.75):
    X_winsorize = winsorize(X, axis = 0, limits = (quantile_inf, 1 - quantile_sup))
    X_rob_scale = pp.robust_scale(X, quantile_range = (100 * quantile_inf, 100 * quantile_sup))
    return X_winsorize, X_rob_scale 
