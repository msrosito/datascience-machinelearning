################################################################################
# Multiclass classification with Gaussian Naive Bayes
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
################################################################################

from sklearn.datasets import load_iris
import sklearn.naive_bayes as snb
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
 

################################################################################
# Function definitions
################################################################################

"""
    gaussian_naivebayes_fit_and_predict_skl(X_train, y_train, X_test = None, 
    y_test = None)

Description
    Gaussian naive Bayes for classification. Train and test.
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix), optional
   `y_test`: test set of true labels (ntest x nfeatures matrix), optional
Outputs
    `mus`: mean of each feature per class
    `sigmas`: variance of each feature per class
    `pred_train_class`: predicted class for each training sample (ntrain array)
    `prior`: probability of each class
    `pred_train_prob`: predicted probabilities for each training sample to
    belong to each class (ntrain x nclasses)
    `pred_test_class`: predicted class for each test sample (ntest array)
    `pred_test_prob`: predicted probabilities for each test sample to
    belong to each class (ntrain x nclasses)
    `accuracy`: mean accuracy on the test set, -1 if y_test is not provided 
"""   

def gaussian_naivebayes_fit_and_predict_skl(X_train, y_train, X_test = None, 
    y_test = None):
    naivebayes_fit = snb.GaussianNB().fit(X_train,y_train)
    mus = naivebayes_fit.theta_
    sigmas = naivebayes_fit.sigma_
    prior = naivebayes_fit.class_prior_
    pred_train_class = naivebayes_fit.predict(X_train)
    pred_train_prob = naivebayes_fit.predict_proba(X_train)
    if X_test != None:
        pred_test_class = naivebayes_fit.predict(X_test)
        pred_test_prob = naivebayes_fit.predict_proba(X_test)
        if y_test != None:
            accuracy = naivebayes_fit.score(X_test, y_test) 
        else:
            accuracy = -1
        return mus, sigmas, prior, pred_train_class, pred_train_prob, \
               pred_test_class, pred_test_prob, accuracy        
    else:
        return mus, sigmas, prior, pred_train_class, pred_train_prob
        
################################################################################
# Main
################################################################################

# Example Wikipedia ############################################################
# See: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
##
print('Example gender according to height, weight, and foot size')

# Data
y = ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F']
X = [[6, 180, 12], [5.92, 190, 11], [5.58, 170, 12], [5.92, 165, 10], 
[5, 100, 6], [5.5, 150, 8], [5.42, 130, 7], [5.75, 150, 9]]
y_test = ['F']
X_test = [[6, 130, 8]]

# Compute Naive Bayes
mus, sigmas, prior, pred_train_class, pred_train_prob, pred_test_class, \
pred_test_prob, accuracy  = gaussian_naivebayes_fit_and_predict_skl(X, y, X_test
, y_test)

# Report results
print('Mean female: ', mus[0])
print('Mean male: ', mus[1])
print('Variance female: ', sigmas[0])
print('Variance male: ', sigmas[1])
print('Prior: ', prior)
print('Predicted classes (train):', pred_train_class)
print('Predicted probabilities (train):', pred_train_prob)
print('Predicted classes (test):', pred_test_class)
print('Predicted probabilities (test):', pred_test_prob)
print('Mean test accuracy: ', accuracy)

# Example iris ################################################################
# See https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
##
print('\n Example flower data set (Iris from scikit-learn ')

# Read data
X, y = load_iris(return_X_y=True, as_frame = True)
X = X.values.tolist()
y = y.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5, 
random_state = 0) # reproducible

# Compute Naive Bayes
mus, sigmas, prior, pred_train_class, pred_train_prob, pred_test_class, \
pred_test_prob, accuracy  = gaussian_naivebayes_fit_and_predict_skl(X_train, 
y_train, X_test, y_test)

# Analysis
confusion_matrix_train = skm.confusion_matrix(y_train, pred_train_class)
report_training = skm.classification_report(y_train, pred_train_class)
report_test = skm.classification_report(y_test, pred_test_class)

# Report results
print('Training set')
print('Means class 0: ', mus[0])
print('Means class 1: ', mus[1])
print('Means class 2: ', mus[2])
print('Prior: ', prior)
print('Confusion matrix: ', confusion_matrix_train)
print('Full report: ', report_training)
print('\n Test set')
print('Accuracy: ', accuracy)
print('Full report: ', report_test)
