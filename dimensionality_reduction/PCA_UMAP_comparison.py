################################################################################
# Dimensionality reduction of a set of images using Principal Component Analysis
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit
################################################################################

import sklearn.decomposition as dsl
import numpy as np
from PIL import Image, ImageOps
import os
import umap
import sklearn.metrics as skm
import sklearn.linear_model as lm

################################################################################
# Function definitions
################################################################################

"""
    principal_components(X):

Description
    Analysis of principal components of a matrix of features using scikit-learn
Input
   `X`: matrix of features (nobs x nfeatures). X must be centered around its mean
   `n_comp`: number of principal components considered
Outputs
    `components`: axis in the feature space
    `exp_variance`: percentage of variance explained by each component
    `T`: projection of each observation in the principal components (nobs x nfeatures)
"""   

def principal_components_skl(X, n_comp):
    pca = dsl.PCA(n_components = n_comp).fit(X)
    components = pca.components_
    ex_variance = pca.explained_variance_ratio_
    T = pca.transform(X)
    return components, ex_variance, T


"""
    images_to_flat_array(img_filename):

Description
    Conversion of an image to a flat array of integers (0 to 255, grayscale)
Input
   `X`: image filename
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
    logistic_regression_fit_and_predict_skl(X_train, y_train, X_test = None, 
    y_test = None, intercept = True)

Description
    Logistic regression for classification. Train and test.
Inputs
   `X_train`: training features (ntrain x nfeatures matrix)
   `y_train`: labels of the true class of each observation (ntrain array)
   `X_test`: test set of features (ntest x nfeatures matrix), optional
   `y_test`: test set of true labels (ntest x nfeatures matrix), optional
   `intercept`: a constant should be added to the fitting (default: True)
Outputs
    `coef`: coefficients
    `inter`: intercept
    `pred_train_class`: predicted class for each training sample (ntrain array)
    `pred_train_prob`: predicted probabilities for each training sample to
    belong to each class (ntrain x nclasses)
    `pred_test_class`: predicted class for each test sample (ntest array)
    `pred_test_prob`: predicted probabilities for each test sample to
    belong to each class (ntrain x nclasses)
    `accuracy`: mean accuracy on the test set  
"""   

def logistic_regression_fit_and_predict_skl(X_train, y_train, X_test = None, 
    y_test = None, intercept = True):
    logreg_fit = lm.LogisticRegression(fit_intercept = intercept).fit(X_train,y_train)
    coef = logreg_fit.coef_
    inter = logreg_fit.intercept_
    pred_train_class = logreg_fit.predict(X_train)
    pred_train_prob = logreg_fit.predict_proba(X_train)
    if X_test != None and y_test != None:
        pred_test_class = logreg_fit.predict(X_test)
        pred_test_prob = logreg_fit.predict_proba(X_test)
        accuracy = logreg_fit.score(X_test, y_test)  
              
        return coef, inter, pred_train_class, pred_train_prob, pred_test_class, 
        pred_test_prob, accuracy        
    else:
        return coef, inter, pred_train_class, pred_train_prob

                   
################################################################################
# Main
################################################################################    
    
output_dir = 'results_dimensionality_reduction/'
input_dir = '../data/Fenix_galaxy_images/'
image_size = 3072

# Read the data and construct the full dataset
X, y = construct_galaxy_dataset(input_dir, image_size)
X = X - X.mean(axis = 0)

# Compute the principal n_comp components
n_comp = 2
c, ev, T_pca = principal_components_skl(X, n_comp)

# Compute the UMAP projection with the same number of components
T_umap = umap.UMAP(n_components = 2, random_state = 0).fit_transform(X)

# Classification of T_pca using logistic regression
c_pc, i_pc, pred_class_pc, pred_prob_pc = logistic_regression_fit_and_predict_skl(T_pca, y)

# Classification of T_umap using logistic regression
c_um, i_um, pred_class_um, pred_prob_um = logistic_regression_fit_and_predict_skl(T_umap, y)

# Report results of both classifications
print('Classification of galaxy images using logistic regression')
labels = ['Non-ELliptical', 'Elliptical']
print('Dimensionality reduction of images using PCA ', n_comp, ' components')
print(skm.classification_report(y, pred_class_pc, target_names = labels))
print('Dimensionality reduction of images using UMAP ', n_comp, ' components')
print(skm.classification_report(y, pred_class_um, target_names = labels))
