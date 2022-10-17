################################################################################
# Multilayer perceptron for vectorial function fitting
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
################################################################################

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# Function definitions
################################################################################


"""
   multilayer_perceptron_regressor_skl(X_train, y_train, X_test, y_test, hls)
   
Description
    Multilayer perceptron for function fitting using scikit-learn
    Loss function: squared error
    Admits early stopping using 15 percent of the test as validation and 0.001 tolerance
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: training function outputs (ntrain x noutput array)
   `X_test`: test features (ntest x nfeatures matrix)
   `y_test`: test function outputs (ntest x noutput array)
   `hls` : list containing the sizes of the hidden layers 
   Default: 2 hidden layers with 100 neurons
Outputs
    `pred_train`: prediction for each training sample (ntrain x noutput array)
    `pred_test`: prediction for each test sample (ntest x noutput array)
    `R2_train`: coefficient of determination for the training set
    `R2_test`: coefficient of determination for the test set
    When using multiple outputs, R2 is the uniform average of R2 of each output
    `loss_curve`: list representing the loss function at each iteration       
"""  

def multilayer_perceptron_regressor_skl(X_train, y_train, X_test, y_test, hls = None):
    if hls == None:
        hls = [100, 100]
    mpreg = MLPRegressor(hidden_layer_sizes = hls, validation_fraction = 0.15, 
    early_stopping = True, tol = 0.001, random_state = 0).fit(X_train, y_train)
    pred_train = mpreg.predict(X_train)
    pred_test = mpreg.predict(X_test)
    R2_train = mpreg.score(X_train, y_train)
    R2_test = mpreg.score(X_test, y_test)
    loss_curve = mpreg.loss_curve_
#    print(mpreg.out_activation_) # identity
    return pred_train, pred_test, R2_train, R2_test, loss_curve
    

"""
   multilayer_perceptron_classifier_tf(X_train, y_train, X_test, y_test, hls)
Description
    Multilayer perceptron for function fitting using tensorflow
    Loss function: squared error
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: training function outputs (ntrain x noutput array)
   `X_test`: test features (ntest x nfeatures matrix)
   `y_test`: test function outputs (ntest x noutput array)
   `hls` : list containing the sizes of the hidden layers
   Default: 2 hidden layers with 100 neurons 
Outputs
    `pred_train`: prediction for each training sample (ntrain x noutput array)
    `pred_test`: prediction for each test sample (ntest x noutput array)
    `R2_train`: coefficient of determination for the training set
    `R2_test`: coefficient of determination for the test set
    When using multiple outputs, R2 is the uniform average of R2 of each output
    `loss_curve_train`: list representing the loss function for the training set at each iteration
    `loss_curve_val`: list representing the loss function for the validation set at each iteration
    `R2_curve_train`: list representing the R2 for the training set at each iteration
    `R2_curve_val`: list representing the R2 for the validation set at each iteration         
"""
    
def multilayer_perceptron_regressor_tf(X_train, y_train, X_test, y_test, hls = None):
    if hls == None:
        hls = [100, 100]
    input_size = len(X_train.iloc[0])
    activation = 'relu'    
    n_hidden_layers = len(hls) 
    model = Sequential(Flatten())
    for i in range(n_hidden_layers):
        model.add(Dense(int(hls[i]), activation = activation)) # hidden layers
    model.add(Dense(2)) # activation = identity
    model.compile(optimizer = 'adam', loss = tf.keras.losses.MeanSquaredError(),
    metrics = tfa.metrics.RSquare())
    history = model.fit(X_train, y_train, batch_size = min(input_size, 200), epochs =  200,
    validation_split = 0.15)
#    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001)    
#    history = model.fit(X_train, y_train, batch_size = min(input_size, 200), epochs =  200, 
#    callbacks = [callback], validation_split = 0.15)
    model.summary()
    loss_curve_train = history.history['loss']
    loss_curve_val = history.history['val_loss']
    R2_curve_train = history.history['r_square']
    R2_curve_val = history.history['val_r_square']
    R2_train = model.evaluate(X_train, y_train)[1]
    R2_test = model.evaluate(X_test, y_test)[1]
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return pred_train, pred_test, R2_train, R2_test, loss_curve_train, loss_curve_val, R2_curve_train, R2_curve_val


"""       
   plot_one_curve(curve, output_dir, label, plot)
Description
   Plot of the loss or R2 curve as a function of the iteration number
Input
    `curve`: values of the curve for each iteration
    `output_dir`: output directory
    `label`: library used to perform the classification 
    `plot`: 'loss' or 'R2' 
""" 
   
def plot_one_curve(curve, output_dir, label, plot):
    n = len(curve)
    filename = output_dir + 'reg-ml-perceptron-' + plot + '-' + label + '.png'
    plt.plot(curve, linewidth = 2)
    plt.title('Multilayer perceptron regressor training ' + label, fontsize = 15.)
    plt.xlabel('Iteration', fontsize = 14)
    plt.ylabel(plot, fontsize = 14)
    plt.text(n * 0.55, max(loss_curve) * 0.8, 'N of iterations: ' + str(n), fontsize = 13)
    plt.savefig(filename)
    plt.close()
    

"""       
   plot_two_curves(train_curve, val_curve, output_dir, label, plot)
Description
   Plot of the loss or R2 curve as a function of the iteration number for both
   the training and validation sets
Input
    `train_curve`: values of the curve for each iteration for the training set
    `val_curve`: values of the curve for each iteration for the validation set
    `output_dir`: output directory
    `label': library used to perform the classification
    `plot`: 'loss' or 'R2'  
""" 
   
def plot_two_curves(train_curve, val_curve, output_dir, label, plot):
    n = len(train_curve)
    filename = output_dir + 'reg-ml-perceptron-train-val' + plot + '-' + label + '.png'
    plt.plot(train_curve, linewidth = 2, label = 'training')
    plt.plot(val_curve, linewidth = 2, label = 'validation', color = 'g')
    plt.title('Multilayer perceptron regressor (training and val) ' + label, fontsize = 14.2)
    plt.legend()
    plt.xlabel('Iteration', fontsize = 14)
    plt.ylabel(plot, fontsize = 14)
    plt.text(n * 0.55, max(train_curve) * 0.8, 'N of iterations: ' + str(n), fontsize = 13)
    plt.savefig(filename)
    plt.close()    
 


################################################################################
# Main
################################################################################

input_dir = '../data/'
output_dir = 'resultsNN/'

# Read the data and construct the training and test datasets
data = pd.read_csv(input_dir + 'data_example_func.csv')
X = data.iloc[:, 0:4] # four features
y = data.iloc[:, -2:] # two outputs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Example using scikit learn
print('Multilayer perceptron regression (scikit-learn)')
# Perform regression
pred_train, pred_test, R2_train, R2_test, loss_curve = multilayer_perceptron_regressor_skl(X_train, 
y_train, X_test, y_test, [15, 15])
# Report results
print('Number of iterations (early-stopping): ', len(loss_curve))
print('R2 (training set): ', R2_train)
print('R2 (test set): ', R2_test)
# Plot loss curve
plot_one_curve(loss_curve, output_dir, 'scikit-learn', 'loss')

# Example using tensorflow
print('\n Multilayer perceptron regression (tensorflow)')
# Perform classification and report the model summary
pred_train, pred_test, R2_train, R2_test, loss_curve_train, loss_curve_val, R2_curve_train, R2_curve_val = multilayer_perceptron_regressor_tf(X_train, y_train, X_test, y_test, [15, 15])
# Report results
print('Number of iterations: ', len(loss_curve_train))
print('R2 (training set): ', R2_train)
print('R2 (test set): ', R2_test)
# plot curves
plot_two_curves(loss_curve_train, loss_curve_val, output_dir, 'tensorflow', 'loss')
plot_two_curves(R2_curve_train, R2_curve_val, output_dir, 'tensorflow', 'R2')                   
