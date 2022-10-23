################################################################################
# Time series analysis using different algorithms
# Goal: predict consumption
#
# Source https://towardsdatascience.com/time-series-modeling-using-scikit-pandas-and-numpy-682e3b8db8d1
# Open Power System Data https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv
################################################################################

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Function definitions
################################################################################

"""
    regression_results(y_true, y_pred)
Description:
    Computation of regression metrics
Inputs:
    `y_true`: true values of the response variable
    `y_pred`: predicted values of the response variable
"""

def regression_results(y_true, y_pred):
    # res = y_true - y_pred 
    # 1 - mean((res - mean(res))**2) / mean((y_true - mean(y))**2)        
    explained_variance = metrics.explained_variance_score(y_true, y_pred) 
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) # mean(|res|)
    mse = metrics.mean_squared_error(y_true, y_pred) # mean((res**2))
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred) 
    # mean((ln(y_true + 1)-ln(y_pred + 1))**2)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred) # median(|res|)
    r2 = metrics.r2_score(y_true, y_pred) # 1 - mean((res**2)) / mean((y_true - mean(y))**2)   
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


"""
    rmse(y_true, y_pred)
Descripton
    Root mean square error (sqrt(mse))
Inputs:
    `y_true`: true values of the response variable
    `y_pred`: predicted values of the response variable
Output:
    `score`: sqrt(mean((res**2)))
"""
   
def rmse(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    res = y_true - y_pred
    square_distance = res**2
    mean_square_distance = square_distance.mean()
    score = mean_square_distance**0.5
    return score


"""
    comparison_of_RFs(X_train, y_train, X_test, y_test, n_trees, max_features, max_depth)
Description:
    Regression using random forest
    Identification of the highest score model
Inputs:
   `X_train`: explanatory variables of the training set
   `y_train`: true values of the response variable of the training set
   `X_test`: explanatory variables of the test set
   `y_test`: true values of the response variable of the training set
   `n_trees`: list of the number of trees in each random forest
   `max_features`: list of the number of features to consider when looking for the best split
   If "auto", then max_features = n_features
   If "sqrt", then max_features = sqrt(n_features)
   If "log2", then max_features = log2(n_features)
   `max_depth`: list of the maximum depths of the trees
Outputs:   
    `best_params`: parameters of the random forest that gave the best results
    `best_score`: mean score of the best_estimator
    `final_train_score`: mean score for the training set after 
    refitting using the best parameters
    `final_test_score`: mean score for the test set after 
    refitting using the best parameters
"""


def comparison_of_RFs(X_train, y_train, X_test, y_test, n_trees, max_features, max_depth):
    params = {'n_estimators' : n_trees, 'max_features' : max_features, 'max_depth' : max_depth}
    estimator = RandomForestRegressor(random_state = 0)
    tscv = TimeSeriesSplit(n_splits=10) # TimeSeries Cross validation
    # the score to validate the performance (rmse) is defined above
    rmse_score = metrics.make_scorer(rmse, greater_is_better = False)
    grid = GridSearchCV(estimator = estimator, param_grid = params, cv = tscv, scoring = rmse_score) 
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_score = abs(grid.best_score_)
    final_train_score = abs(grid.score(X_train, y_train))
    final_test_score = abs(grid.score(X_test, y_test))
    return best_params, best_score, final_train_score, final_test_score

################################################################################
# Main
################################################################################

output_dir = 'results/'

# Read dataset
url='https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
data = pd.read_csv(url) 

data['Date'] = pd.to_datetime(data['Date']) # convert the date column to type DATETIME
data = data.set_index('Date')

# Baseline model: predict consumption based on (1) yesterday's consumption and 
# (2) the difference between yesterday and the day before yesterday’s consumption 
# new dataframe
data_consumption = data[['Consumption']] # new dataframe
# new column with yesterday's consumption values
data_consumption.loc[:,'Yesterday'] = data_consumption.loc[:,'Consumption'].shift()
# new column with difference between yesterday and day before yesterday's consumption values
data_consumption.loc[:, 'Yesterday_Diff'] = data_consumption.loc[:, 'Yesterday'].diff()
data_consumption.dropna(inplace = True) # drop rows which have missing values

# Definition of training (2006-2016) and test set (2017)
X_train = data_consumption.loc[data_consumption.index.year <= 2016].drop(['Consumption'], axis = 1)
y_train = data_consumption.loc[data_consumption.index.year <= 2016, 'Consumption']
X_test = data_consumption.loc[data_consumption.index.year == 2017].drop(['Consumption'], axis = 1)
y_test = data_consumption.loc[data_consumption.index.year == 2017, 'Consumption']

# Comparison of different models
models = []
models.append(('LR', LinearRegression())) # linear regression
models.append(('NN', MLPRegressor(random_state = 0, solver = 'lbfgs')))  # multilayer perceptron
models.append(('KNN', KNeighborsRegressor())) # 5 nearest neighbors
models.append(('RF', RandomForestRegressor(n_estimators = 10, random_state = 0))) # random forest
models.append(('SVR', SVR(gamma='auto'))) # support vector regression - kernel = linear

tscv = TimeSeriesSplit(n_splits=10) # TimeSeries Cross validation
# Example n_samples = 6, n_splits = 5
# TRAIN: [0] VAL: [1]
# TRAIN: [0 1] VAL: [2]
# TRAIN: [0 1 2] VAL: [3]
# TRAIN: [0 1 2 3] VAL: [4]
# TRAIN: [0 1 2 3 4] VAL: [5]

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    cv_results = cross_val_score(model, X_train, y_train, cv = tscv, scoring = 'r2')
    print('Model: ', name, '- Mean CV score: ', np.round(cv_results.mean(), 5), 
    '- Standard deviation: ', np.round(cv_results.std(), 5))
    names.append(name)
    results.append(cv_results)

# Box plot
plt.boxplot(results, labels = names)
plt.title('Algorithm Comparison')
plt.savefig(output_dir + 'comparison-box-plot.png')
plt.close()

# Hyperparameter tuning: the best random forest
best_params, best_score, final_train_score, final_test_score = comparison_of_RFs(X_train, 
y_train, X_test, y_test, n_trees = [20, 50, 100], max_features = ['auto', 'sqrt', 'log2'], 
max_depth = [i for i in range(5, 10)])
n_trees_best = best_params['n_estimators']
max_features_best = best_params['max_features']
max_depth_best = best_params['max_depth']

# Report results of the hyperparameter tuning
print('Number of estimators of the best RF model: ', n_trees_best)
print('Maximum features to consider in each split in the best RF model: ', max_features_best)
print('Maximum depth of trees in the best RF model: ', max_depth_best)
print('Best training rmse: ', best_score)
print('Best training rmse after refitting: ', final_train_score)
print('Best test rmse: ', final_test_score)

# Check the best random forest performance
best_model = RandomForestRegressor(n_estimators = n_trees_best, max_features = max_features_best,
max_depth = max_depth_best, random_state = 0).fit(X_train, y_train)
print('Regression metrics for the best random forest (training set)')
regression_results(y_train, best_model.predict(X_train))
print('Regression metrics for the best random forest (test set)')
regression_results(y_test, best_model.predict(X_test))
