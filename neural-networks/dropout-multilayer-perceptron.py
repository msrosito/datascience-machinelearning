################################################################################
# Multilayer perceptron for multiclass classification using dropout regulatization
#
# See: https://www.tensorflow.org/api_docs/python/tf/keras/models
#      https://www.tensorflow.org/api_docs/python/tf/keras/layers
# Dataset: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#
################################################################################
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


################################################################################
# Function definitions
################################################################################
    
    
"""
   multilayer_perceptron_classifier_dropout_tf(X_train, y_train, X_test, y_test, 
   hls = None, drop_ratios = None)
Description
    Multilayer perceptron for classification using tensorflow with dropout regularization
    before each layer
    Loss function: cross entropy. 
Inputs
   `X_train`: training set of features (ntrain x nfeatures matrix)
   `y_train`: training set of true labels ((ntest, nclasses))
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels ((ntest, nclasses))
   `hls` : list containing the sizes of the hidden layers 
   Default: 2 hidden layers with 100 neurons
   `drop_ratios`: list containing the dropout ratios before each layer (n_hidden_layers + 1)
    Default: [0., 0., 0.]
Outputs
    `pred_train_class`: predicted probabities for each training sample (ntrain, nclasses)
    `pred_test_class`: predicted probabilities for each test sample (ntest, nclasses)
    `accuracy_train`: accucary for the training set
    `accuracy_test`: accucary for the test set
    `lc_train`: list representing the loss function for the training set at each iteration
    `lc_val`: list representing the loss function for the validation set at each iteration
    `ac_train`: list representing the accuracy for the training set at each iteration
    `ac_val`: list representing the accuracy for the validation set at each iteration        
""" 
   
def multilayer_perceptron_classifier_dropout_tf(X_train, y_train, X_test, y_test, hls = None, drop_ratios = None):
    if hls == None:
        hls = [100, 100]
    n_hidden_layers = len(hls)     
    if drop_ratios == None:
        drop_ratios = [0.] * (n_hidden_layers + 1)
    input_size = len(X_train[0])
    n = len(X_train)
    activation = 'relu'    
    model = Sequential(Flatten())
    for i in range(n_hidden_layers):
        model.add(Dropout(float(drop_ratios[i])))
        model.add(Dense(int(hls[i]), activation = activation)) # hidden layers
    model.add(Dropout(float(drop_ratios[n_hidden_layers])))
    model.add(Dense(3, activation = 'softmax')) 
    model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
    metrics = 'accuracy')
    history = model.fit(X_train, y_train, batch_size = n, epochs =  300, validation_split = 0.15)
    model.summary()
    lc_train = history.history['loss']
    lc_val = history.history['val_loss']
    ac_train = history.history['accuracy']
    ac_val = history.history['val_accuracy']
    accuracy_train = model.evaluate(X_train, y_train)[1]
    accuracy_test = model.evaluate(X_test, y_test)[1]
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return pred_train, pred_test, accuracy_train, accuracy_test, lc_train, lc_val, ac_train, ac_val

    
"""       
   plot_two_curves(train_curve, val_curve, output_dir, drop, plot)
Description
   Plot of the loss or accuracy curve as a function of the iteration number for both
   the training and validation sets
Input
    `train_curve`: values of the curve for each iteration for the training set
    `val_curve`: values of the curve for each iteration for the validation set
    `output_dir`: output directory
    `drop': 'no-dropout' or 'dropout'
    `plot`: 'loss' or 'accuracy'  
""" 
   
def plot_two_curves(train_curve, val_curve, output_dir, drop, plot):
    n = len(train_curve)
    filename = output_dir + drop + 'ml-perceptron-train-val' + plot + '.png'
    plt.plot(train_curve, linewidth = 2, label = 'training')
    plt.plot(val_curve, linewidth = 2, label = 'validation', color = 'g')
    plt.title('Multilayer perceptron classifier (training and val) ' + drop, fontsize = 14.2)
    plt.legend()
    plt.xlabel('Iteration', fontsize = 14)
    plt.ylabel(plot, fontsize = 14)
    plt.text(n * 0.55, max(train_curve) * 0.65, 'N of iterations: ' + str(n), fontsize = 13)
    plt.savefig(filename)
    plt.close() 


################################################################################
# Main
################################################################################

output_dir = 'resultsNN/'
tf.keras.utils.set_random_seed(3)

# Read the data and construct the training and test datasets
iris = load_iris()
X = iris['data'] # 4 features
y = iris['target'] # 3 classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Perform classifications
# Without dropout
hls = [100, 10, 100]
pred_train, pred_test, accuracy_train, accuracy_test, lc_train, lc_val, ac_train, ac_val = multilayer_perceptron_classifier_dropout_tf(X_train, y_train, X_test, y_test, hls = hls)
print(accuracy_train, accuracy_test)
# Using dropout
ratios = [0.25, 0.25, 0.2, 0.]
pred_train_d, pred_test_d, accuracy_train_d, accuracy_test_d, lc_train_d, lc_val_d, ac_train_d, ac_val_d = multilayer_perceptron_classifier_dropout_tf(X_train, y_train, X_test, y_test, hls = hls, drop_ratios = ratios)

# Report results
print('Multilayer perceptron without dropout')
print('Accuracy (training set): ', accuracy_train)
print('Accuracy (test set): ', accuracy_test)
print('Multilayer perceptron using dropout')
print('Accuracy (training set): ', accuracy_train_d)
print('Accuracy (test set): ', accuracy_test_d)

# Plots
plot_two_curves(lc_train, lc_val, output_dir, 'no-dropout', 'loss')
plot_two_curves(ac_train, ac_val, output_dir, 'no-dropout', 'accuracy')
plot_two_curves(lc_train_d, lc_val_d, output_dir, 'dropout', 'loss')
plot_two_curves(ac_train_d, ac_val_d, output_dir, 'dropout', 'accuracy')      
