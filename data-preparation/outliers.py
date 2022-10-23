################################################################################
# Process data to deal with outliers
#
# See: https://scikit-learn.org/stable/modules/preprocessing.html
#      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html
# Outliers creation: https://www.geeksforgeeks.org/winsorization/
################################################################################

import matplotlib.pyplot as plt
import datapreparation as dp
import numpy as np

################################################################################
# Function definitions
################################################################################

"""
    create_outliers(array, n_low, n_high)
Description
    Addition of n_low + n_high ouliers to a given array
Inputs
    `array`: 1D numpy array
    `n_low`: number of outliers on the lower end
    `n_high`: number of outliers on the upper end
"""

def create_outliers(array, n_low, n_high):
    AlreadySelected = [] # the values which are selected for creating outliers 
    # are appended so that same outliers are not created again 
    mean = np.mean(array)
    i = 0
    while(i < n_low):
        x = np.random.choice(array)
        y = x - mean * 3
        if y not in AlreadySelected:
            AlreadySelected.append(y)
            i += 1
        else:
            continue
    i = 0            
    while(i < n_high):
        x = np.random.choice(array)
        y = x + mean * 4
        if y not in AlreadySelected:
            AlreadySelected.append(y)
            i += 1
        else:
            continue            
    return np.concatenate((array, AlreadySelected))

    
"""
   print_statistics(array, quantile_inf, quantile_sup)
Description
    Displays the mean, standard deviation, median, and IQR of a given array
    IQR is defined as quantile_sup - quantile_inf.
    If quantile_sup = 0.75 and quantile_inf =  0.25, IQR depicts the interquartile range.
Inputs
    `array`: 1D numpy array
    `quantile_inf`: lower limit for IQR
    `quantile_sup`: upper limit for IQR
"""        
def print_statistics(array, quantile_inf, quantile_sup):
    print('Mean: ', np.mean(array))
    print('Standard dev: ', np.std(array))
    print('Median: ', np.median(array))
    print('IQR (quantil_sup - quantil_inf): ', 
    np.percentile(array, 100 * quantile_sup) - np.percentile(array, 100 * quantile_inf))
    
        
################################################################################
# Main
################################################################################

output_dir = 'results/'

# Example 1

# Input features matrix
A = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [10,9,8,7,6]])

# Transformation
winsorized, robust_scale = dp.data_preprocessing_with_outliers(A, 0.25, 0.75)

# Results
print('Simple example')
print('Original array: ', A)
print('Winsorized array: ', winsorized)
print('Robustly scaled array: ', robust_scale)

# Example 2

np.random.seed(seed=0)

# Array generation
array = [np.random.randint(100) for i in range(100)]
array = create_outliers(array, 5, 5)

# Transformation
qinf = 0.2
qsup = 0.8
winsorized, robust_scale = dp.data_preprocessing_with_outliers(array, qinf, qsup)

# Report results
print('\n Example: robustness to outliers')
print('Original array')
print_statistics(array, qinf, qsup)
print('\n Winsorized array')
print_statistics(winsorized, qinf, qsup)
print('\n Robustly scaled array')
print_statistics(robust_scale, qinf, qsup)

# Box plots
filename1 = 'original_box_plot.png' # original
plt.boxplot(array)
plt.title('Array with outliers')
plt.savefig(output_dir + filename1)
plt.close()

filename2 = 'winsorized_box_plot.png' # winsorized array
plt.boxplot(winsorized)
plt.title('Winsorized array')
plt.savefig(output_dir + filename2)
plt.close()
