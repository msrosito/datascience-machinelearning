################################################################################
# Evaluation of binary classifiers using scikit learn
#
# See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
################################################################################

from evaluate_classification import *

################################################################################
# Main
################################################################################

output_dir = 'results_classifiers/'
input_dir = '../data/'

# Simple example ###############################################################
filename_out = output_dir + 'simple_example.png'

# Definition of variables
y = np.array([0, 0, 1, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8])

# Values for the ROC curve
fpr, tpr, thresholds = skm.roc_curve(y, scores)

# Computation of the AUC and plot
auc = roc_plot_and_auc(y, scores, filename_out, 'Example 1')

# Confusion matrix from the scores with a threshold = 0.5
y_predicted, M = confusion_matrix_from_scores(y, scores, 0.5)

# Report results
print('True values: ',y)
print('Predicted values: ',y_predicted)
print('Confusion matrix:', M)
print('Area under curve: ', auc)

# ROC of a random classifier ###################################################
filename_out = output_dir + 'random_example.png'

# Definition of variables
y = np.array([0, 0, 0, 1, 1])
scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

# Values for the ROC curve
fpr, tpr, thresholds = skm.roc_curve(y, scores)
print(fpr, tpr, thresholds)

# Computation of the AUC and plot
auc = roc_plot_and_auc(y, scores, filename_out, 'Random classifier')

# Report area under curve
print('Area under curve: ', auc)

# Another example  #############################################################
filename_in = input_dir + 'data_classification.csv'    
filename_out = output_dir + 'another_example.png'

# Read true labels and scores
y_true, y_score = read_vars(filename_in)

# Compute predicted values and confusion matrix with threshold = 0.5
y_predicted, M = confusion_matrix_from_scores(y_true, y_score, 0.5)

# Compute AUC and plot the ROC curve
auc = roc_plot_and_auc(y_true, y_score, filename_out, 'Example 2')

# Report results
print('Confusion matrix:', M)
print('Area under curve: ', auc)
