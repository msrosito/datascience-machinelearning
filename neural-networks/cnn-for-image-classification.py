################################################################################
# Convolutional Neural Network for the binary classification of galaxy images
#
# See: https://www.tensorflow.org/api_docs/python/tf/keras/layers
#      https://www.tensorflow.org/api_docs/python/tf/keras/models
################################################################################

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt

################################################################################
# Function definitions
################################################################################


"""
    images_to_matrix(img_filename):
Description
    Conversion of an image to a matrix of integers (0 to 255, grayscale)
Input
   `img_filename`: image filename
Output
    `mat`: numpy 2D array (height x width)
"""   
    
def images_to_matrix(img_filename):
    img = Image.open(img_filename) # (height, width, nchannels)
    img = ImageOps.grayscale(img)  # (height, width) 
    mat = np.asarray(img)
    return mat


"""
    construct_galaxy_dataset(input_dir, size)
Description
    Construct the arrays of features and labels for the full galaxy dataset 
Inputs
   `input_dir`: directory containing the images of Non-Elliptical (0) 
   and Elliptical (1) galaxies
   `h`: height of each input image
   `w`: width of each input image
Outputs
    `X_feat`: 3D matrix containing the images 
    (n_non-ellipticals + n_ellpiticals, h, w)
    `y_label`: label of each sample (1 if elliptical, 0 otherwise)
"""        
    
def construct_galaxy_dataset(input_dir, h, w):
    dir0 = input_dir + 'Non-Elliptical/'
    dir1 = input_dir + 'Elliptical/'
    files0 = os.listdir(dir0)
    nfiles0 = len(files0)
    files1 = os.listdir(dir1)
    nfiles1 = len(files1)
    y_label = np.zeros(nfiles0 + nfiles1)
    y_label[nfiles0:] = 1
    X_feat = np.zeros((nfiles0 + nfiles1, h, w))
    i = 0
    for name in files0:
        img_filename = str(dir0) + name
        imgarr = images_to_matrix(img_filename)
        X_feat[i] = imgarr
        i += 1
    for name in files1:
        img_filename = str(dir1) + name
        imgarr = images_to_matrix(img_filename)
        X_feat[i] = imgarr
        i += 1
    return X_feat, y_label


"""
   cnn_classifier_tf(X_train, y_train, X_test, y_test, n_filters, size_kernels)
Description
    Convolutional neural network for the classification of images using tensorflow
    Loss function: log-loss
Inputs
   `X_train`: training set of features (ntrain x img_height x img_width array)
   `y_train`: training set of true labels (ntrain)
   `X_test`: test set of features (ntrain x img_height x img_width array)
   `y_test`: test set of true labels (ntest)
   `n_filters`: list containing the number of filters in each convolutional layer
   `size_kernels`: list containing the kernel sizes for each convolutional layer
Outputs
    `pred_train_class`: predicted probabities for each training sample (ntrain array)
    `pred_test_class`: predicted probabilities for each test sample (ntest array)
    `accuracy_train`: accuracy for the training set
    `accuracy_test`: accuracy for the test set
    `lc_train`: list representing the loss function for the training set at each iteration
    `lc_val`: list representing the loss function for the validation set at each iteration
    `ac_train`: list representing the accuracy for the training set at each iteration
    `ac_val`: list representing the accuracy for the validation set at each iteration      
""" 
   
def cnn_classifier_tf(X_train, y_train, X_test, y_test, n_filters, size_kernels):
    n = len(X_train)
    input_height, input_width = np.shape(X_train[0])
    print(input_height, input_width)
    n_conv_layers = len(n_filtrers) 
    activation = 'relu'
    model = Sequential([
        # convolutional layer
        Conv2D(n_filtrers[0], size_kernels[0], padding = 'same', 
        activation = activation, input_shape = (input_height, input_width, 1)),
        MaxPooling2D() # pooling layer
        ])
    for i in range(1, n_conv_layers):
        model.add(Conv2D(n_filtrers[i], size_kernels[i], padding = 'same', 
        activation = activation))
        MaxPooling2D()
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(20, activation = activation))     
    model.add(Dense(1, activation = 'sigmoid')) 
    model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
    metrics = 'accuracy')
    history = model.fit(X_train, y_train, batch_size = n // 3, epochs =  30, validation_split = 0.2)
    model.summary()
    lc_train = history.history['loss']
    lc_val = history.history['val_loss']
    ac_train = history.history['accuracy']
    ac_val = history.history['val_accuracy']    
    accuracy_train = model.evaluate(X_train, y_train)[1]
    accuracy_test = model.evaluate(X_test, y_test)[1]
    pred_train_prob = model.predict(X_train)
    pred_test_prob = model.predict(X_test)
    return pred_train_prob, pred_test_prob, accuracy_train, accuracy_test, lc_train, lc_val, ac_train, ac_val


"""       
   plot_two_curves(train_curve, val_curve, output_dir, plot)
Description
   Plot of the loss or accuracy curve as a function of the iteration number for both
   the training and validation sets
Input
    `train_curve`: values of the curve for each iteration for the training set
    `val_curve`: values of the curve for each iteration for the validation set
    `output_dir`: output directory
    `plot`: 'loss' or 'accuracy'  
""" 
   
def plot_two_curves(train_curve, val_curve, output_dir, plot):
    n = len(train_curve)
    filename = output_dir + 'cnn-classifier-train-val-' + plot + '.png'
    plt.plot(train_curve, linewidth = 2, label = 'training')
    plt.plot(val_curve, linewidth = 2, label = 'validation', color = 'g')
    plt.title('CNN image classifier (training and val) ', fontsize = 14.2)
    plt.legend()
    plt.xlabel('Iteration', fontsize = 14)
    plt.ylabel(plot, fontsize = 14)
    plt.text(n * 0.55, max(train_curve) * 0.8, 'N of iterations: ' + str(n), fontsize = 13)
    plt.savefig(filename)
    plt.close()    


################################################################################
# Main
################################################################################

output_dir = 'resultsNN/'
input_dir = '../data/Fenix_galaxy_images/'
img_height = 48
img_width = 64
tf.keras.utils.set_random_seed(0)

# Read the data and construct the training and test datasets
X, y = construct_galaxy_dataset(input_dir, img_height, img_width)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Perform classification
n_filtrers = [8, 4, 8]
size_kernels = [[3, 3], [3 ,3], [3, 3]]
pred_train_prob, pred_test_prob, accuracy_train, accuracy_test, lc_train, lc_val, ac_train, ac_val = cnn_classifier_tf(X_train, y_train, X_test, y_test, n_filtrers, size_kernels)

# Report results
print('CNN for image classification')
print('Accuracy (training set): ', accuracy_train)
print('Accuracy (test set): ', accuracy_test)

# Plots
plot_two_curves(lc_train, lc_val, output_dir, 'loss')
plot_two_curves(ac_train, ac_val, output_dir, 'accuracy')
