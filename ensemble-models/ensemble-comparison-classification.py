################################################################################
# Comparison between three ensemble models and a single decision tree for
# binary classification
#
# Dataset: https://github.com/rahul-raoniar/Rahul_Raoniar_Blogs
################################################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn.neighbors as knn
import sklearn.linear_model as lm
from sklearn import tree  
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import pandas as pd


################################################################################
# Function definitions
################################################################################

"""
    random_forest_skl(n_estimators, max_samples, X_train, y_train, X_test = None, 
    y_test = None)

Description
    Random Forest for classification using boostrap. Train and test using entropy.
Inputs
   `nt`: number of trees
   `max_samples`: number of samples of X_train to train each tree
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix), optional
   `y_test`: test set of true labels (ntest array), optional
Outputs
    `f_importance`: impurity based features importance
    `pred_train_class`: predicted class for each training sample (ntrain array)
    `pred_test_class`: predicted class for each test sample (ntest array)
    `pred_train_prob`: mean predicted class probability of the trees for each training sample (ntrain array)
    `pred_test_prob`: mean predicted class probability of the trees for each test sample (ntest array)
    `accuracy`: mean accuracy on the test set, -1 if y_test is not provided  
""" 
def random_forest_classification_skl(nt, max_samples, X_train, y_train, 
    X_test = None, y_test = None):
    random_forest = RandomForestClassifier(n_estimators = nt, max_samples = max_samples,
    criterion = 'entropy', random_state = 0).fit(X_train, y_train) # reproducible
    f_importance = random_forest.feature_importances_
    
    pred_train_class = random_forest.predict(X_train)
    pred_train_prob = random_forest.predict_proba(X_train)
    if X_test != None:
        pred_test_class = random_forest.predict(X_test)
        pred_test_prob = random_forest.predict_proba(X_test)
        if y_test != None:
            accuracy = random_forest.score(X_test, y_test) 
        else:
            accuracy = -1
                  
        return f_importance, pred_train_class, pred_train_prob, pred_test_class, pred_test_prob, accuracy
    else:
        return f_importance, pred_train_class, pred_train_prob


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
   `y_test`: test set of true labels (ntest array), optional
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


"""
    myensemble_skl(X_train, y_train, X_test, y_test = None)

Description
    Ensemble model for binary classification. ALgorithms: 3-nearest neighbors, 
    decision tree, logistic regression
    
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels (ntest array)
   
Outputs
    `y_test_labels_c1`: 3-nn predicted class for each test sample (ntest array) 
    `y_test_labels_c2`: decision tree predicted class for each test sample (ntest array)
    `y_test_labels_c3`: logistic regression predicted class for each test sample (ntest array)
    `final_test_labels`: ensemble predicted class for each test sample         
"""
def myensemble_skl(X_train, y_train, X_test, y_test):
    # first classifier
    knn3 = knn.KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
    print('First classifier')
    print(skm.classification_report(y_train, knn3.predict(X_train)))
    # second classifier 
    dtc = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 0).fit(X_train, y_train)
    print('Second classifier')
    print(skm.classification_report(y_train, dtc.predict(X_train)))
    # third classifier
    lr = lm.LogisticRegression(max_iter = 300).fit(X_train,y_train)
    print('Third classifier')
    print(skm.classification_report(y_train, lr.predict(X_train))) # ver en que falla cada una
    # predictions
    y_test_labels_c1 = knn3.predict(X_test)
    y_test_labels_c2 = dtc.predict(X_test)
    y_test_labels_c3 = lr.predict(X_test)
    final_test_labels = (y_test_labels_c1 + y_test_labels_c2 + y_test_labels_c3) / 3
    # final labels
    m = len(y_test)
    for i in range(m):
        if final_test_labels[i] < 0.5:
            final_test_labels[i] = 0
        else:
            final_test_labels[i] = 1
    return y_test_labels_c1, y_test_labels_c2, y_test_labels_c3, final_test_labels

  
"""           
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
"""        

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

print('Classification using random forest (scikit-learn)')

# Read the data
filename_in = input_dir + 'diabetes.csv'    
data = pd.read_csv(filename_in)
data['diabetes'] = data['diabetes'].map({'neg': 0, 'pos': 1})
X_train, y_train, X_test, y_test = csv_to_train_and_test(data, 0.2) # test = 20 percent
feature_names = data.columns
nf = len(feature_names) - 1

# Compute random forest
nt = 10
ms = 150
f_importance1, pred_train_class1, pred_train_prob1, pred_test_class1, pred_test_prob1, accuracy1  = random_forest_classification_skl(nt, ms, X_train, y_train, X_test, y_test)

# Compute the boosted ensemble
ne = 10
f_importance2, estimator_weights2, pred_train_class2, pred_train_prob2, pred_test_class2, pred_test_prob2 = adaboost_decisiontree_skl(ne,
X_train, y_train, X_test, y_test)

# Compute ensemble 3-nn + decision tree + logistic
y_pred1, y_pred2, y_pred3, final_pred = myensemble_skl(X_train, y_train, X_test, y_test)

# Analysis
M_random_forest = skm.confusion_matrix(y_test, pred_test_class1)
M_adaboost = skm.confusion_matrix(y_test, pred_test_class2)
M_myensemble = skm.confusion_matrix(y_test, final_pred)

# Report results
print('Random forest classification report')
print(M_random_forest)
print('AdaBoost forest classification report')
print(M_adaboost)
print('My ensemble forest classification report')
print(M_myensemble)
