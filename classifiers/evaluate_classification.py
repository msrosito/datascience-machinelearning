################################################################################
# Evaluation of binary classifiers using scikit learn
#
# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
################################################################################

import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import csv

################################################################################
# Function definitions
################################################################################

"""
    binary_classifier_evaluation_metrics(M)

Description
    Evaluation of a binary classifier using the confusion matrix
Input
   `M`: confusion matrix (2 x 2) :M_ij : number of elements in class i 
   predicted to be in class j.
Outputs
    `presicion`: TP / (TP + FP)
    `sensitivity or recall`: TP / (TP + FN)
    `specificity`: TN / (TN + FP)
    `negative predicted value`: TN / (TN + FN)
    `accuracy`: (TP + TN) / (TP + TN + FP + FN)
    `F1`: F1 score
"""
def binary_classifier_evaluation(M):
    n0_true = M[0][0] + M[0][1]
    n0_pred = M[0][0] + M[1][0]
    n1_true = M[1][1] + M[1][0]
    n1_pred = M[1][1] + M[0][1]
    n_tot = M[0][0] + M[0][1] + M[1][0] + M[1][1]

    pres = M[1][1] / n1_pred
    sens = M[1][1] / n1_true
    spec = M[0][0] / n0_true
    npv = M[0][0] / n0_pred
    acc = (M[1][1] + M[0][0]) / n_tot
    F1 = 2 * pres * sens / (pres + sens)
    
    return pres, sens, spec, npv, acc, F1
    
    
"""
    confusion_matrix_from_scores(y_true, y_score, threshold):

Description
    Computation of the confusion matrix given the decision function from a classifier
Inputs
   `y_true`: true labels (0 and 1)
   `y_score`: probability estimates of the positive class (decision function)
   'threshold`: the elements above this value are classified as positive
Outputs
   `y_predicted:`: predicted labels (0 and 1)
   `M`: confusion matrix (2 x 2) :M_ij : number of elements in class i 
   predicted to be in class j.
"""
def confusion_matrix_from_scores(y_true, y_score, threshold):
    n = len(y_score)
    y_predicted = np.zeros(n)
    y_predicted[y_score >= threshold] = 1
    M = skm.confusion_matrix(y_true, y_predicted)
    return y_predicted, M   
    
    
"""
    roc_plot_and_auc(y_true, y_score)

Description
    Plot of the ROC curve and computation of the area under curve (AUC)
    Use for binary classifiers
Inputs
   `y_true`: true labels (0 and 1, or -1 and 1)
   `y_score`: probability estimates of the positive class (decision function)
   `filename`: filename
Output
    `AUC`: Area under the ROC curve
"""    
def roc_plot_and_auc(y_true, y_score, filename, label):
    false_positive_rate, true_positive_rate, thresholds = skm.roc_curve(y_true, y_score)
    auc = skm.roc_auc_score(y_true, y_score)
    plt.plot(false_positive_rate, true_positive_rate, c = 'b', linewidth = 2)
    plt.plot([0.,0.5, 1.], [0., 0.5, 1.], '--', c = 'r', 
    label = 'Random classifier')
    plt.plot([0], [1], 'o', markersize = 12, c = 'g', label = 'Perfect classifier')
    fs = 16
    plt.xticks(fontsize = 0.75 * fs)
    plt.yticks(fontsize = 0.75 * fs)
    plt.title('ROC space '+ str(label), fontsize = fs)
    plt.text(0.2, 0.01, 'AUC: '+ str(round(auc, 3)), fontsize = fs)
    plt.xlabel('False positive rate: FP/(TN + FP)', fontsize = fs)
    plt.ylabel('True positive rate: TP/(TP + FN)', fontsize = fs)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    
    return auc


"""
    read_vars(filename)
   TODO: complete this
"""
def read_vars(filename):
    n = 100 # TODO: fix this
    y_true = np.zeros(n)
    y_score = np.zeros(n)
    i = 0
    with open(filename) as data_file:
        csvreader = csv.reader(data_file)
        next(csvreader)
        for row in csvreader:
            y_true[i] = row[0]
            y_score[i] = row[1]
            i += 1    
    
    return y_true, y_score
