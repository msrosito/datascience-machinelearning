################################################################################
# Multilayer perceptron for the binary classification of galaxy images
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
#      https://www.tensorflow.org/api_docs/python/tf/keras/layers
#      https://www.tensorflow.org/api_docs/python/tf/keras/models
################################################################################

from sklearn.neural_network import MLPClassifier
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt


################################################################################
# Function definitions
################################################################################
 

"""
    images_to_flat_array(img_filename):
Description
    Conversion of an image to a flat array of integers (0 to 255, grayscale)
Input
   `img_filename`: image filename
Outputs
    `arr`: flat numpy array (height x width)
    `shape`: shape of the original image in pixels (height, width)
"""   
    
def images_to_flat_array(img_filename):
    img = Image.open(img_filename)
    img = ImageOps.grayscale(img)
    arr = np.asarray(img)
    shape_img = np.shape(arr)
    arr = arr.flatten()
    return arr, shape_img
 
    
"""
    construct_galaxy_dataset(input_dir, size)
Description
    Construct the arrays of features and labels for the full galaxy dataset 
Inputs
   `input_dir`: directory containing the images of Non-Elliptical (0) 
   and Elliptical (1) galaxies
   `size`: number of pixels of each image
Outputs
    `X_feat`: numpy matrix of flattened images of the galaxies 
    (n_non-ellipticals + n_ellpiticals, size)
    `y_label`: label of each sample (1 if elliptical, 0 otherwise)
"""        
    
def construct_galaxy_dataset(input_dir, size):
    dir0 = input_dir + 'Non-Elliptical/'
    dir1 = input_dir + 'Elliptical/'
    files0 = os.listdir(dir0)
    nfiles0 = len(files0)
    files1 = os.listdir(dir1)
    nfiles1 = len(files1)
    y_label = np.zeros(nfiles0 + nfiles1)
    y_label[:nfiles0] = 0
    y_label[nfiles0:] = 1
    X_feat = np.zeros((nfiles0 + nfiles1, size))
    i = 0
    for name in files0:
        img_filename = str(dir0) + name
        imgarr = images_to_flat_array(img_filename)[0]
        X_feat[i] = imgarr
        i += 1
    for name in files1:
        img_filename = str(dir1) + name
        imgarr = images_to_flat_array(img_filename)[0]
        X_feat[i] = imgarr
        i += 1
    return X_feat, y_label


"""
   multilayer_perceptron_classifier_skl(X_train, y_train, X_test, y_test, hls   
Description
    Multilayer perceptron for classification using scikit-learn
    Loss function: log-loss 
Inputs
   `X_train`: training set of features (ntrain x nfeatures matrix)
   `y_train`: training set of true labels (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels (ntest array)
   `hls` : list containing the sizes of the hidden layers 
   Default: 2 hidden layers with 100 neurons
Outputs
    `pred_train_class`: predicted class for each training sample (ntrain array)
    `pred_test_class`: predicted class for each test sample (ntest array)
    `accuracy_train`: mean accucary for the training set
    `accuracy_test`: mean accucary for the test set
    `loss_curve`: list representing the loss function at each iteration       
"""    

def multilayer_perceptron_classifier_skl(X_train, y_train, X_test, y_test, hls = None):
    if hls == None:
        hls = [100, 100]
    mpclass = MLPClassifier(hidden_layer_sizes = hls, random_state = 0).fit(X_train, y_train)
    pred_train_class = mpclass.predict(X_train)
    pred_test_class = mpclass.predict(X_test)
    accuracy_train = mpclass.score(X_train, y_train)
    accuracy_test = mpclass.score(X_test, y_test)
    loss_curve = mpclass.loss_curve_
#    print(mpclass.out_activation_) # logistic
    return pred_train_class, pred_test_class, accuracy_train, accuracy_test, loss_curve
    
    
"""
   multilayer_perceptron_classifier_tf(X_train, y_train, X_test, y_test, hls)
Description
    Multilayer perceptron for classification using tensorflow
    Loss function: log-loss. 
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: training set of true labels (ntest array)
   `X_test`: test set of features (ntest x nfeatures matrix)
   `y_test`: test set of true labels (ntest array)
   `hls` : list containing the sizes of the hidden layers 
   Default: 2 hidden layers with 100 neurons
Outputs
    `pred_train_class`: predicted probabities for each training sample (ntrain array)
    `pred_test_class`: predicted probabilities for each test sample (ntest array)
    `accuracy_train`: mean accucary for the training set
    `accuracy_test`: mean accucary for the test set
    `loss_curve`: list representing the loss function at each iteration   
    
"""    
def multilayer_perceptron_classifier_tf(X_train, y_train, X_test, y_test, hls = None):
    if hls == None:
        hls = [100, 100]
    input_size = len(X_train[0])
    activation = 'relu'    
    n_hidden_layers = len(hls) 
    model = Sequential(Flatten())
    for i in range(n_hidden_layers):
        model.add(Dense(int(hls[i]), activation = activation)) # hidden layers
    model.add(Dense(1, activation = 'sigmoid')) 
    model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
    metrics = 'accuracy')
    history = model.fit(X_train, y_train, batch_size = min(input_size, 200), epochs =  200, validation_split = 0)
    model.summary()
    loss_curve = history.history['loss']
    accuracy_train = model.evaluate(X_train, y_train)[1]
    accuracy_test = model.evaluate(X_test, y_test)[1]
    pred_train_prob = model.predict(X_train)
    pred_test_prob = model.predict(X_test)
    return pred_train_prob, pred_test_prob, accuracy_train, accuracy_test, loss_curve

                  
"""       
   plot_loss_curve(output_dir, loss_curve, label)
Description
   Plot of the loss curve as a function of the iteration number
Input
    `loss_curve`: values of the loss curve for each iteration
    `output_dir`: output directory
    `label': library used to perform the classification 
""" 
   
def plot_loss_curve(loss_curve, output_dir, label):
    n = len(loss_curve)
    filename = output_dir + 'bin-class-ml-perceptron-' + label + '.png'
    plt.plot(loss_curve, linewidth = 2)
    plt.title('Multilayer perceptron classifier training ' + label, fontsize = 15.5)
    plt.xlabel('Iteration', fontsize = 14)
    plt.ylabel('Loss', fontsize = 14)
    plt.text(n * 0.55, max(loss_curve) * 0.8, 'N of iterations: ' + str(n), fontsize = 13)
    plt.savefig(filename)
    plt.close()
             
    
################################################################################
# Main
################################################################################

output_dir = 'resultsNN/'
input_dir = '../data/Fenix_galaxy_images/'
image_size = 3072
tf.keras.utils.set_random_seed(3)


# Read the data and construct the training and test datasets
X, y = construct_galaxy_dataset(input_dir, image_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Example using scikit learn
print('Multilayer perceptron binary classification (scikit-learn)')
# Perform classification
pred_train_class, pred_test_class, accuracy_train, accuracy_test, loss_curve = multilayer_perceptron_classifier_skl(X_train, 
y_train, X_test, y_test, [15, 15])
# Report results
print('Number of iterations: ', len(loss_curve))
print('Mean accuracy (training set): ', accuracy_train)
print('Classification summary (training set):')
print(skm.classification_report(y_train, pred_train_class))
print('Mean accuracy (test set): ', accuracy_test)
print('Classification summary (test set):')
print(skm.classification_report(y_test, pred_test_class))
# Plot loss curve
plot_loss_curve(loss_curve, output_dir, 'scikit-learn')

# Example using tensorflow
print('\n Multilayer perceptron binary classification (tensorflow)')
# Perform classification and report the model summary
pred_train_prob, pred_test_prob, accuracy_train, accuracy_test, loss_curve = multilayer_perceptron_classifier_tf(X_train, 
y_train, X_test, y_test, [15, 15])
# Report results
print('Number of iterations: ', len(loss_curve))
print('Mean accuracy (training set): ', accuracy_train)
print('Mean accuracy (test set): ', accuracy_test)
print(pred_train_prob)
# Plot loss curve
plot_loss_curve(loss_curve, output_dir, 'tensorflow')
