################################################################################
# Evaluation of binary classification with logistic regression
#
# See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# Dataset: https://github.com/rahul-raoniar/Rahul_Raoniar_Blogs
################################################################################

import sklearn.linear_model as lm
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluate_classification import *

################################################################################
# Function definitions
################################################################################

"""
    logistic_regression_fit_and_predict_skl(X_train, y_train, X_test = None, 
    y_test = None, intercept = True)

Description
    Logistic regression for classification. Train and test.
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix), optional
   `y_test`: test set of true labels (ntest array), optional
   `intercept`: a constant should be added to the fitting (default: True)
Outputs
    `coef`: coefficients
    `inter`: intercept
    `pred_train_class`: predicted class for each training sample (ntrain array)
    `pred_train_prob`: predicted probabilities for each training sample to
    belong to each class (ntrain x nclasses)
    `pred_test_class`: predicted class for each test sample (ntest array)
    `pred_test_prob`: predicted probabilities for each test sample to
    belong to each class (ntest x nclasses)
    `accuracy`: mean accuracy on the test set  
"""   

def logistic_regression_fit_and_predict_skl(X_train, y_train, X_test = None, 
    y_test = None, intercept = True):
    logreg_fit = lm.LogisticRegression(fit_intercept = intercept).fit(X_train,y_train)
    coef = logreg_fit.coef_
    inter = logreg_fit.intercept_
    pred_train_class = logreg_fit.predict(X_train)
    pred_train_prob = logreg_fit.predict_proba(X_train)
    if X_test != None and y_test != None:
        pred_test_class = logreg_fit.predict(X_test)
        pred_test_prob = logreg_fit.predict_proba(X_test)
        accuracy = logreg_fit.score(X_test, y_test)  
              
        return coef, inter, pred_train_class, pred_train_prob, pred_test_class, 
        pred_test_prob, accuracy        
    else:
        return coef, inter, pred_train_class, pred_train_prob
        

################################################################################
# Main
################################################################################

output_dir = 'results_classifiers/'
input_dir = '../data/'

# Read the data
data = pd.read_csv(input_dir + 'diabetes.csv')
data['diabetes'] = data['diabetes'].map({'neg': 0, 'pos': 1})

X = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])

# Example train only ###########################################################
print('Classification using logistic regrssion (scikit-learn), train only')
# Compute the logistic regression
coef, inter, pred_labels, probabilities = logistic_regression_fit_and_predict_skl(X, y)

# Evaluation
# Confusion matrix (threshold 0.5, as used in scikit learn function)
filename = output_dir + 'ROC_logistic_example.png'
y_p, M = confusion_matrix_from_scores(y, probabilities.T[1], 0.5)
M2 = skm.confusion_matrix(y, pred_labels)
AUC = roc_plot_and_auc(y, probabilities.T[1], filename, 'Logistic regression')
pres, sens, spec, npv, acc, F1 = binary_classifier_evaluation(M)

# Report results
print('Intercept: ', inter)
print('Coefficients: ', coef)
print('Confusion Matrix: ', M)
print(M2)
print('Area under curve ROC: ', AUC)
print('Presicion TP / (TP + FP): ', pres)
print('Recall TP / (TP + FN): ', sens)
print('Specificity TN / (TN + FP): ', spec)
print('Negative predicted value TN / (TN + FN): ', npv)
print('Accuracy (TP + TN) / (TP + TN + FP + FN): ', acc)
print('F1 Score: ', F1)
