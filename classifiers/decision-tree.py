################################################################################
# Binary classification with Decision Trees
#
# See: https://scikit-learn.org/stable/modules/tree.html#classification
# Dataset: https://github.com/rahul-raoniar/Rahul_Raoniar_Blogs
################################################################################

from evaluate_classification import *
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

################################################################################
# Function definitions
################################################################################

"""
    decision_tree_classification_skl(X_train, y_train, X_test = None, 
    y_test = None, intercept = True)

Description
    Decision trees for classification. Train and test using entropy.
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix), optional
   `y_test`: test set of true labels (ntest array), optional
Outputs
    `decision_tree`: decision tree
    `depth`: tree depth
    `leaves`: number of leaves
    `pred_train_class`: predicted class for each training sample (ntrain array)
    `pred_test_class`: predicted class for each test sample (ntest array)
    `accuracy`: mean accuracy on the test set, -1 if y_test is not provided  
""" 
def decision_tree_classification_skl(X_train, y_train, X_test = None,
    y_test = None):
    decision_tree = tree.DecisionTreeClassifier(criterion = 'entropy', 
    random_state = 0).fit(X_train, y_train)

    depth = decision_tree.get_depth()
    leaves = decision_tree.get_n_leaves()
    pred_train_class = decision_tree.predict(X_train)
    if X_test != None:
        pred_test_class = decision_tree.predict(X_test)
        if y_test != None:
            accuracy = decision_tree.score(X_test, y_test) 
        else:
            accuracy = -1
                    
        return decision_tree, depth, leaves, pred_train_class, pred_test_class, accuracy
    else:
        return decision_tree, depth, leaves, pred_train_class
        

'''
    plot_decision_tree(df, filename, title, feature_names = None, 
    class_names = None)
    
    Description:
    Plot of a decision tree
    Inputs
        `df`: decision tree estimator
        `filename`: filename of the plot
        `title`: title for the plot
        `features_names`: names of the features, optional
        `class_names`: labels for the target variable, optional
'''
def plot_decision_tree(df, filename, title, feature_names = None, class_names = None):
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(15)
    tree.plot_tree(dt, feature_names = feature_names, 
    class_names = class_names, filled = True, impurity = True)
    plt.title(title)
    plt.savefig(filename) 
    
        
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, 
    random_state = 0) # reproducible
    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()
    X_test = X_test.values.tolist()
    y_test = y_test.values.tolist()
    return X_train, y_train, X_test, y_test

        
################################################################################
# Main
################################################################################

output_dir = 'results_classifiers/'
input_dir = '../data/' 

print('Classification using decision trees (scikit-learn)')

# Read the data
filename_in = input_dir + 'diabetes.csv'    
filename_out = output_dir + 'decision_tree_example'
data = pd.read_csv(filename_in)
data['diabetes'] = data['diabetes'].map({'neg': 0, 'pos': 1})
X_train, y_train, X_test, y_test = csv_to_train_and_test(data, 0.5)

# Compute the decision tree
dt, depth, leaves, pred_train_class, pred_test_class, accuracy  = \
decision_tree_classification_skl(X_train, y_train, X_test, y_test)

# Analysis
M_train = skm.confusion_matrix(y_train, pred_train_class)
pres_tr, sens_tr, spec_tr, npv_tr, acc_tr, F1_tr = binary_classifier_evaluation(M_train)
M_test = skm.confusion_matrix(y_test, pred_test_class)
pres_te, sens_te, spec_te, npv_te, acc_te, F1_te = binary_classifier_evaluation(M_test)

# Report results
print('Training set')
print('Confusion matrix: ', M_train)
print('Presicion TP / (TP + FP): ', pres_tr)
print('Recall TP / (TP + FN): ', sens_tr)
print('Specificity TN / (TN + FP): ', spec_tr)
print('Negative predicted value TN / (TN + FN): ', npv_tr)
print('Accuracy (TP + TN) / (TP + TN + FP + FN): ', acc_tr)
print('F1 Score: ', F1_tr)
print('Test set')
print('Confusion matrix: ', M_test)
print('Presicion TP / (TP + FP): ', pres_te)
print('Recall TP / (TP + FN): ', sens_te)
print('Specificity TN / (TN + FP): ', spec_te)
print('Negative predicted value TN / (TN + FN): ', npv_te)
print('Accuracy (TP + TN) / (TP + TN + FP + FN): ', acc_te)
print('F1 Score: ', F1_te)

# Plot
fn = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass',
'pedigree', 'age']
classes = ['Negative', 'Positive']
title = 'Decision tree diabetes'
plot_decision_tree(dt, filename_out, title, feature_names = fn, class_names = classes)
