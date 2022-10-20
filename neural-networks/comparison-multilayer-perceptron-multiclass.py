################################################################################
# Multilayer perceptron for multiclass classification
# Performance comparison using different activation functions and solvers
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
#      https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# Dataset: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#
################################################################################

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV

################################################################################
# Function definition
################################################################################

"""
    comparison_of_classifiers(X_train, y_train, X_test, y_test, hls, 
    activation_list, solver_list)
Description:
    Classification problem using multilayer perceptron with different activation 
    functions and solvers
    Identification of the highest score model
Inputs:
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels (ntest array)
   `activation_list`: list of activation functions
   `solver_list`: list of solvers
   `hls` : list containing the sizes of the hidden layers 
   Default: 2 hidden layers with 100 neurons
Outputs:   
    `best_params`: parameters of the model that gave the best results
    `best_score`: mean cross-validated score (mean accuracy) of the best_estimator
    `results`: dictionary containing a summary for each set of parameters
    `final_train_score`: mean accucary for the training set after 
    refitting using the best parameters
    `final_test_score`: mean accucary for the test set after 
    refitting using the best parameters
"""

def comparison_of_classifiers(X_train, y_train, X_test, y_test, activation_list, 
    solver_list, hls = None):
    if hls == None:
        hls = [100, 100]
    params = {'activation' : activation_list, 'solver' : solver_list}
    estimator = MLPClassifier(hidden_layer_sizes = hls, random_state = 0, tol = 0.002)
    grid = GridSearchCV(estimator = estimator, param_grid = params, cv = 2)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_score = grid.best_score_
    results = grid.cv_results_
    final_train_score = grid.score(X_train, y_train)
    final_test_score = grid.score(X_test, y_test)
    return best_params, best_score, results, final_train_score, final_test_score


################################################################################
# Main
################################################################################    

# Read the data and construct the training and test datasets
iris = load_iris()
X = iris['data'] # 4 features
y = iris['target'] # 3 classes
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)

# Classification using different solvers and activation functions
best_params, best_score, results, final_train_score, final_test_score = comparison_of_classifiers(X_train, y_train, 
X_test, y_test, activation_list = ['identity', 'logistic'], solver_list = ['lbfgs', 'adam', 'sgd'], hls = [10, 10])

# Report results
print('Activation function of the best model: ', best_params['activation'])
print('Solver of the best model: ', best_params['solver'])
print('Best training score: ', best_score)
print('Best training score after refitting: ', final_train_score)
print('Best test score: ', final_test_score)
print('Mean scores for each set of parameters:')
for i, par in enumerate(results['params']):
    print('Parameters: ', par)
    print(results['mean_test_score'][i])
