# [WIP] Data Science & Machine Learning Examples

This repository contains basic python code examples about different methods of data science and machine learning. This is a work in progress.

Current examples

## Data preparation
    - Missing values: simple and k-nearest neighbors imputation (`scikit-learn`), row removal.
    - Data preprocessing and outliers (`scikit-learn` and `scipy.stats`): centering, standarizing, normalization, robust scaling, winsorization.
    - Log-transformations and Box-Cox (`scipy.stats`) transformation to deal with outliers.
    - Examples.
    - Plots: 2D plots (scatter, lines, heatmaps), 3D plots (scatter), histograms.
    
## Linear models
    - Simple and multiple regression using `scikit-learn` and `statsmodels` libraries.
    - Generalized linear model Poisson using `scikit-learn`.
    - Linear regression with correlated residuals. Generalized least squares using `statsmodels`.
    - Regularization of linear models with L1 and L1 using `scikit-learn`.
    - Comparison between a multiple linear regression and a decision tree using `scikit-learn`.
    
## Classifiers
    - Evaluation of binary classifiers using `scikit-learn`.
    - Logistic regression using `scikit-learn`.
    - Naive Bayes using `scikit-learn`.
    - Decision tree using `scikit-learn`. 
    - Cross-validation and stratified cross-validation using `scikit-learn`.
    
## Dimensionality reduction
    - Principal component analysis using `scikit-learn`.
    - Comparison bewteen classificaiton of images reduced by PCA and UMAP algorithms.
    
## Clustering
    - K-means, DBSCAN, and HBSCAN comparison: times and plots using `scikit-learn`
    
## Ensemble models
    - Binary classification with Random Forest using `scikit-learn`.
    - Binary classification with AdaBoost using decision trees using `scikit-learn`.
    - Comparison between random forest, adaboost, and an ensemble formed by 3-nearest neighbors, logistic regression and decision tree.
    - Regression using Random Forest using `scikit-learn`. Comparison with linear regression.
    
## Neural networks
    - Convolutional neural network using `tensorflow`. See also [here](https://github.com/msrosito/deep-learning-galaxy).
    - Multilayer perceptron for the binary classification using `scikit-learn` and `tensorflow`.
    - Multilayer perceptron for vectorial function fitting using `scikit-learn` and `tensorflow`.
    - Selection of the best multilayer perceptron varying hyperparameters for multiclass classification using `scikit-learn`.
    - Dropout regularization on multilayer perceptron for multiclass classification using `tensorflow`.
    
## Time series
    - Time series regression using `scikit-learn`.
    
## SQL & DB design (MySQL examples)
    - example1.sql: subqueries, JOIN, GROUP BY, aggregation functions, filtrering, ORDER BY.
    - example2.sql: subqueries, GROUP BY, aggregation functions, filtrering.
    - example3.sql: Common Table Expressions (CTE), window functions, filtrering.
    - example4.sql: reusing calculations, JOIN, GROUP BY, aggregation functions, IF.
    - example5.sql: EXISTS, filtrering, extract from datetime, subqueries, GROUP BY, aggregation functions.
    
## Dataframes (Python pandas)
    - example1.py: define and read dataframes, access data, add data, dataframe summary.
    - example2.py: merge, reshape, and concatenate dataframes.
    - example3.py: filtrer and group data, window functions. 
    - example4.py and example5.py: exercises.
    
