################################################################################
# Dimensionality reduction of a set of images using Principal Component Analysis
#
# See: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit
################################################################################

import sklearn.decomposition as dsl
import numpy as np
from PIL import Image, ImageOps
import os

################################################################################
# Function definitions
################################################################################

"""
    principal_components_skl(X):

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
n_comp = 10
c, ev, T = principal_components_skl(X, n_comp)

# Report the results
print('Porcentage of variance explained by each component: ')
for i in range(n_comp):
    print('PC' + str(i + 1), ': ', np.round(ev[i], 3))
print('Total explained variance using ', n_comp, ' components: ', np.sum(ev))

exit()

