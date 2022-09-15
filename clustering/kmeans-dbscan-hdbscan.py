################################################################################
# Comparison of times between K-means, DBSCAN, and HDBSCAN clustering algoritmhs
#
# See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
#     https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
#     https://github.com/scikit-learn-contrib/hdbscan
################################################################################

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
from matplotlib import colors

from sklearn.cluster import DBSCAN, KMeans
import hdbscan

################################################################################
# Function definitions
################################################################################

"""
    plot_clusters(x, y, l):

Description
    2D scatter plot of the data colored according the clustering
Inputs
   `x`: x-coordinate
   `y`: y-coordinate
   `l`: cluster labels.
   First cluster = 0, l <= 0 means outlier
   `algorithm`: clustering algorithm
   `output_dir`: output directory
"""
def plot_clusters(x, y, l, algorithm, output_dir):
    Nc = max(l) + 1
    clustered = (l >= 0) # removes outliers
    x1 = x[clustered]
    y1 = y[clustered]
    l1 = l[clustered]
    cmapq = plt.cm.rainbow
    norm = colors.BoundaryNorm(np.arange(-0.5, Nc + 0.5, 1), cmapq.N)    
    sc = plt.scatter(x1, y1, c = l1, s = 15, cmap = cmapq, norm = norm, edgecolor='none')
    plt.colorbar(sc, ticks = np.arange(Nc + 1))
    plt.xlabel('x', fontsize = 14)
    plt.ylabel('y', fontsize = 14)
    plt.title(algorithm + ' Clustering, N clusters: ' + str(Nc), fontsize = 15)
    plt.savefig(output_dir + algorithm + ' Clustering.png')
    plt.close()


################################################################################
# Main
################################################################################

input_dir = '../data/'
output_dir = 'results-clustering/'
filename = input_dir + '2D-proj.txt'

# Read the projection coordinates

projection = np.loadtxt(filename)
x = projection[:, 0]
y = projection[:, 1]

# Plot

plt.plot(x, y, linewidth = 0, marker = 'o', markersize = 3)
plt.xlabel("x", fontsize = 14)
plt.ylabel("x", fontsize = 14)
plt.title('Unclustered data', fontsize = 15)
plt.savefig(output_dir + 'unclustered.png')
plt.close()

# HDBSCAN parameters

nn = 70 
ms =  1 
mcs = 70
md = 0

# Example of clustering using HDBSCAN ##########################################
print('Clustering using HDBSCAN')

# Compute clustering
t0 = time.time()
clusters = hdbscan.HDBSCAN().fit(projection)
t1 = time.time()
Nc_h =  clusters.labels_.max() + 1
l = clusters.labels_
nsamples = len(x)
out = nsamples - len(x[l >= 0])

# Report result
print('Number of clusters: ', str(Nc_h))
print('Number of samples: ', str(nsamples))
print('Number of outliers: ', str(out))
print('Time: ', t1 - t0)

# Plot results
plot_clusters(x, y, l, 'HDBSCAN', output_dir)

# Example of clustering using DBSCAN ##########################################
print('Clustering using DBSCAN')

# Compute clustering
t0 = time.time()
clusters = DBSCAN().fit(projection)
t1 = time.time()
Nc_d =  clusters.labels_.max() + 1
l = clusters.labels_
nsamples = len(x)
out = nsamples - len(x[l >= 0])

# Report result
print('Number of clusters: ', str(Nc_d))
print('Number of samples: ', str(nsamples))
print('Number of outliers: ', str(out))
print('Time: ', t1 - t0)

# Plot results
plot_clusters(x, y, l, 'DBSCAN', output_dir)

# Example of clustering using k-means ##########################################
print('Clustering using k-means')

# Compute clustering with the same number of  clusters than HDBSCAN
t0 = time.time()
clusters = KMeans(Nc_h).fit(projection)
t1 = time.time()
l = clusters.labels_
nsamples = len(x)
out = nsamples - len(x[l >= 0])

# Report result
print('Number of clusters: ', str(Nc_h))
print('Number of samples: ', str(nsamples))
print('Time: ', t1 - t0)

# Plot results
plot_clusters(x, y, l, 'K-means (K HDBSCAN)', output_dir)
