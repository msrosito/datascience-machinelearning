################################################################################
# Binary classification with AdaBoost using decision trees
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
# Dataset: https://github.com/rahul-raoniar/Rahul_Raoniar_Blogs
################################################################################

from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import pandas as pd

################################################################################
# Function definitions
################################################################################

"""
    adaboost_decisiontree_skl(n_estimators, max_samples, X_train, y_train, X_test = None, 
    y_test = None)

Description
    AdaBoost for classification using decision trees. Train and test using entropy.
Inputs
   `n_estimators`: maximum number of estimators
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix), optional
   `y_test`: test set of true labels (ntest x nfeatures matrix), optional
Outputs
    `f_importance`: impurity based features importance
    `estimator_weights`: weights for each estimator in the ensemble
    `pred_train_class`: predicted class for each training sample (ntrain array)
    `pred_test_class`: predicted class for each test sample (ntest array)
    `pred_train_prob`: mean predicted class probability of the trees for each training sample (ntrain array)
    `pred_test_prob`: mean predicted class probability of the trees for each test sample (ntest array)
""" 
def adaboost_decisiontree_skl(n_estimators, X_train, y_train, X_test = None, y_test = None):
    base = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 0) # reproducible
    adaboost = AdaBoostClassifier(base_estimator = base, 
    n_estimators = n_estimators, random_state = 0, learning_rate = 0.000000000000001).fit(X_train, y_train) # reproducible
    f_importance = adaboost.feature_importances_
    estimator_weights = adaboost.estimator_weights_
    
    pred_train_class = adaboost.predict(X_train)
    pred_train_prob = adaboost.predict_proba(X_train)
    if X_test != None:
        pred_test_class = adaboost.predict(X_test)
        pred_test_prob = adaboost.predict_proba(X_test)
                  
        return f_importance, estimator_weights, pred_train_class, pred_train_prob, pred_test_class, pred_test_prob
    else:
    
        return f_importance, estimator_weights, pred_train_class, pred_train_prob

            
'''
    csv_to_train_and_test(data, test_size)
    
    Description
        Conversion of a dataset to sets of training and test
    Inputs
        `data`: dataframe
        `test_size`: fraction of the data belonging to the test set
    Outputs
        X_train: training features
        y_train: training target (last column)
        X_test: test features
        y_test: test target (last column)        
'''
def csv_to_train_and_test(data, test_size):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, 
    random_state = 0) # reproducible
    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()
    X_test = X_test.values.tolist()
    y_test = y_test.values.tolist()
    return X_train, y_train, X_test, y_test

        
################################################################################
# Main
################################################################################

input_dir = '../data/' 

print('Classification using AdaBoost decision trees (scikit-learn)')

# Read the data
filename_in = input_dir + 'diabetes.csv'    
data = pd.read_csv(filename_in)
data['diabetes'] = data['diabetes'].map({'neg': 0, 'pos': 1})
X_train, y_train, X_test, y_test = csv_to_train_and_test(data, 0.5) # test = 50 percent
feature_names = data.columns
nf = len(feature_names) - 1

# Compute the boosted ensemble
ne = 10
f_importance, estimator_weights, pred_train_class, pred_train_prob, pred_test_class, pred_test_prob = adaboost_decisiontree_skl(ne,
X_train, y_train, X_test, y_test)
nest = len(estimator_weights[estimator_weights != 0])

# Analysis
M_train = skm.confusion_matrix(y_train, pred_train_class)
M_test = skm.confusion_matrix(y_test, pred_test_class)

# Report results
print('Feature importance: ')
for i in range(nf):
    print(feature_names[i], str(round(f_importance[i], 3) * 100), '%')
print('Number of weak learners: ', nest)
print('Training set summary:')
print(skm.classification_report(y_train, pred_train_class))
print('Test set summary:')
print(skm.classification_report(y_test, pred_test_class))
