################################################################################
# Transformation of skewed data
#
# See: https://scikit-learn.org/stable/modules/preprocessing.html
#      https://scikit-learn.org/stable/modules/impute.html
#      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html
################################################################################

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import datapreparation as dp

################################################################################
# Function definitions
################################################################################

"""
    plot_histograms(data_before, data_after, output_dir, skewness, label)
Description
    Comparison plot of two histograms: skewed data and transformed data
Inputs
    `data_before`: array containing skewed data
    `data_after`: transformed array
    `output_dir`: output directory
    `label`: 'right-skewed', 'right-skewed', 'non-gaussian-right', 'non-gaussian-left'
    Choose non-gaussian if Box-Cox transformation was used
"""

def plot_histograms(data_before, data_after, output_dir, label):
    filename = output_dir + 'histogram-before-after' + label
    if label == 'right-skewed' or label == 'left-skewed':
        transformation = 'log-transformation'
    elif label == 'non-gaussian-right' or label == 'non-gaussian-left':
        transformation = 'box-cox'
    else:
        print('Choose a valid option for label')
        return
    plt.hist(data_before, label = label, color = 'b', density = True, histtype = 'step')
    plt.hist(data_after, label = transformation, color = 'r', density = True, histtype = 'step')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    
    
"""
    plot_boxplot(data_before, transformed, winsorized, output_dir, label)
Description
    Comparison plot of two boxplots: skewed data, transformed data, and winsorized data
Inputs
    `data_before`: array containing skewed data
    `transformed`: log-transformed array
    `winsorized`: winsorized array
    `output_dir`: output directory
    `label`: 'right-skewed', 'left-skewed'
"""

def plot_boxplot(data_before, transformed, winsorized, output_dir, label):
    filename = output_dir + 'boxplot-before-after' + label
    labels = [label, 'log-transformation', 'winsorized']
    all_data = [data_before, transformed, winsorized]
    plt.boxplot(all_data, labels = labels)
    plt.savefig(filename)
    plt.close()    


################################################################################
# Main
################################################################################    

output_dir = 'results/dist/'
np.random.seed(0) 

# Generation of skewed arrays
right_skewed_array = np.random.beta(2, 8, 1000) # upper end outliers
left_skewed_array = np.random.beta(8, 2, 1000) # low end outliers
skewness_right = stats.skew(right_skewed_array)
skewness_left = stats.skew(left_skewed_array)

# log-transformations
log_transformation_right = np.log(right_skewed_array + 0.1)
log_transformation_left = np.log(max(left_skewed_array + 0.1) - left_skewed_array)
skewness_log_right = stats.skew(log_transformation_right)
skewness_log_left = stats.skew(log_transformation_left)
# Winsorization
winsorized_right = dp.data_preprocessing_with_outliers(right_skewed_array)[0]
winsorized_left = dp.data_preprocessing_with_outliers(left_skewed_array)[0]
# Box Cox
boxcox_right, lambda_right = stats.boxcox(right_skewed_array)
boxcox_left, lambda_left = stats.boxcox(left_skewed_array)

# Report results
print('Skewness of the right skewed array: ', skewness_right)
print('Skewness of the log-transformed right skewed array: ', skewness_log_right)
print('Lambda parameter for the Box-Cox transformation of the right skewed array: ', lambda_right)
print('Skewness of the left skewed array: ', skewness_left)
print('Skewness of the log-transformed left skewed array: ', skewness_log_left)
print('Lambda parameter for the Box-Cox transformation of the left skewed array: ', lambda_left)

# Histograms
plot_histograms(right_skewed_array, log_transformation_right, output_dir, 'right-skewed')
plot_histograms(right_skewed_array, log_transformation_right, output_dir, 'non-gaussian-right')
plot_histograms(left_skewed_array, log_transformation_left, output_dir, 'left-skewed')
plot_histograms(left_skewed_array, log_transformation_right, output_dir, 'non-gaussian-left')
# Box plots
plot_boxplot(right_skewed_array, log_transformation_right, winsorized_right, output_dir, 'right-skewed')
plot_boxplot(left_skewed_array, log_transformation_left, winsorized_left, output_dir, 'left-skewed')
