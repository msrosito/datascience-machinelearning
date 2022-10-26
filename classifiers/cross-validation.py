################################################################################
# Comparison between cross validation and stratified cross validation
#
# See https://scikit-learn.org/stable/modules/cross_validation.html
# Dataset: https://github.com/rahul-raoniar/Rahul_Raoniar_Blogs
################################################################################

import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
import pandas as pd

################################################################################
# Function definitions
################################################################################

"""
    cross_validation_split_analysis(X, y, k, stratified = False)
    
Description
    Report of the sizes and proportions of positive samples in each fold for cross-validation    
Inputs
    `X `: training features (nsamples x nfeatures matrix or dataframe) 
    `y`: labels of the true class of each observation (nsamples)
    `class_names`: names for each class (postive 1 and negative 0)
    `stratified`: if true, StratifiedKFold cross-validator, if false KFold cross-validator.
    Default: False
"""

def cross_validation_split_analysis(X, y, class_names, k, stratified = False):
    pos = class_names[1] # positive class
    if stratified:
        split = ms.StratifiedKFold(n_splits = k).split(X, y)
    else:
        split = ms.KFold(n_splits = 5).split(X, y)
    for train_index, test_index in split:
        y_train = y[train_index]
        y_test = y[test_index]
        n_train = len(y_train)
        n_test = len(y_test)
        npos_train = len(y_train[y_train == 'pos'])
        npos_test = len(y_test[y_test == 'pos'])
        print('Positives in the training set: ' , round(npos_train / n_train * 100., 2), '% of a total of ', n_train)  
        print('Positives in the test set: ' , round(npos_test / n_test * 100., 2), '% of a total of ', n_test)
                

################################################################################
# Main
################################################################################

input_dir = '../data/'

# Read the data
df = pd.read_csv(input_dir + 'diabetes.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print('Total sample size: ', len(X))
print('Number of positives: ', len(df[df.diabetes == 'pos']))

# Analysis of the splits 
k = 5 # number of folds
class_names = ['neg', 'pos']

print('Cross-validation', k, '-fold')
cross_validation_split_analysis(X, y, class_names, k)

print('Stratified cross-validation', k, '-fold')
cross_validation_split_analysis(X, y, class_names, k, stratified = True)

# Analysis of the scores of the logistic regression in each case
lr = LogisticRegression(max_iter = 300)
kfold = ms.KFold(n_splits = k)
skfold = ms.StratifiedKFold(n_splits = k)
sk1 = ms.cross_val_score(lr, X, y, cv = kfold)
sk2 = ms.cross_val_score(lr, X, y, cv = skfold) # default if the estimator is a classifier

print('Scores for each run of the k-fold cross-validation: ' , sk1)
print('Scores for each run of the stratified k-fold cross-validation: ', sk2)
