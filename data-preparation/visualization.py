import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


output_dir = 'visual/'
cmap = plt.cm.rainbow
np.random.seed(0)

## 2D plots ####################################################################

# 2D scatter plots

# Data generation
size = 140
nclass = 3
xmin, xmax = -1., 1.
ymin, ymax = -2., 2.
x = np.linspace(xmin, xmax, size)
y = np.random.rand(size)
z1 = x ** 2 + y ** 2 # continous
z2 = np.random.randint(0, nclass, size) # categorical
# scatter plot colored by a third continous variable
plt.scatter(x, y, c = z1, cmap = cmap, marker = 'o', s = 15)
plt.xticks(np.linspace(-1, 1, 5))
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label = 'z')
plt.title('2D scatter plot - continous variable')
plt.savefig(output_dir + '2D-scatter-continous.png')
plt.close()
# scatter plot colored by a third caterogical variable (colorbar)
norm = colors.BoundaryNorm(np.arange(-0.5, nclass + 0.5, 1), cmap.N)
plt.scatter(x, y, c = z2, cmap = cmap, norm = norm)
plt.xticks(np.linspace(-1, 1, 5))
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label = 'z', ticks = np.arange(0, nclass))
plt.title('2D scatter plot - categorical variable')
plt.savefig(output_dir + '2D-scatter-categorical-colorbar.png')
plt.close()
# scatter plot colored by a third caterogical variable (legend)
colors = ['b', 'r', 'g']
labels = ['a', 'b', 'c']
for i in range(nclass):
    in_class_i = z2 == i
    plt.plot(x[in_class_i], y[in_class_i], marker = 'o', linewidth = 0, 
    label = labels[i], color = colors[i])
plt.legend()
plt.title('2D scatter plot - categorical variable')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(output_dir + '2D-scatter-categorical-legend.png')
plt.close()

# 2D lineplots

# Data generation
size = 15
x = np.linspace(-1, 1)
y1 = x / (x ** 2 + 1) + 2
y2 = np.exp(x)
ymin = np.amin([y1, y2])
ymax = np.amax([y1, y2])
# simple plot
plt.plot(x, y1, linewidth = 1.2, color = 'b', label = 'func 1')
plt.plot(x, y2, linewidth = 1.2, marker = '.', color = 'r', label = 'func 2')
plt.title('Simple 2D line plot')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(ymin, ymax)
plt.legend()
plt.savefig(output_dir + '2D-lineplot-legend.png')
plt.close()
# plot with filled region
top = y1 + 0.2
bottom = y1 - 0.1
plt.plot(x, y1, linewidth = 1.2, color = 'k')
plt.plot(x, top, linewidth = 1.2, color = 'k', linestyle = 'dotted')
plt.plot(x, bottom, linewidth = 1.2, color = 'k', linestyle = 'dotted')
plt.fill_between(x, y1, top, facecolor = 'm', alpha = 0.5)
plt.fill_between(x, y1, bottom, facecolor = 'm', alpha = 0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(output_dir + '2D-lineplot-filled.png')
plt.close()

# 2D heatmap

# Data generation
nrows, ncols = 50, 50
xmin, xmax = -1., 1.
ymin, ymax = -2., 2.
# Grid
x, y = np.meshgrid(np.linspace(xmin, xmax, nrows), np.linspace(ymin, ymax, ncols))
z = x ** 2 + y ** 2
z = z[:-1, :-1]
plt.pcolormesh(x, y, z, cmap = cmap, vmin = np.amin(z), vmax = np.amax(z))
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.colorbar(label = 'z')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(output_dir + '2D-heatmap.png')
plt.close()


## 3D plots ####################################################################

# 3D scatter plots

# Data generation
size = 200
nclass = 3
xmin, xmax = -1., 1.
ymin, ymax = -2., 2.
zmin, zmax = 0.5, 1.
nclass = 3
x = np.linspace(xmin, xmax, size)
y = np.linspace(ymin, ymax, size)
z = np.random.uniform(zmin, zmax, size)
w1 = x + y + np.log(z) # continous
w2 = np.random.randint(0, nclass, size) # categorical
# simple scatter plot
ax = plt.axes(projection = '3d') ##### 3D
ax.scatter(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('3D scatter plot')
plt.savefig(output_dir + '3D-scatter.png')
plt.close()
# scatter plot coloured by a continous fourth variable
fig = plt.figure()
ax = plt.axes(projection = '3d') ##### 3D
sc = ax.scatter(x, y, z, c = w1, cmap = cmap)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.colorbar(sc, label = 'z')
plt.title('3D scatter plot - continous variable')
plt.savefig(output_dir + '3D-scatter-continous-color.png')
plt.close()
# scatter plot with different marker sizes according to a continous fourth variable
s = 15 * np.abs(w1) 
ax = plt.axes(projection = '3d') ##### 3D
ax.scatter(x, y, z, s = s, color = 'm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('3D scatter plot - continous variable')
plt.savefig(output_dir + '3D-scatter-continous-sizes.png')
plt.close()
# scatter plot coloured by a fourth caterogical variable (legend)
colors = ['b', 'r', 'g']
labels = ['a', 'b', 'c']
markers = ['o', '^', 's']
ax = plt.axes(projection = '3d') ##### 3D
for i in range(nclass):
    in_class_i = w2 == i
    ax.scatter(x[in_class_i], y[in_class_i], z[in_class_i], color = colors[i], marker = markers[i], label = labels[i])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.title('3D scatter plot - categorical variable')
plt.savefig(output_dir + '3D-scatter-categorical-markers.png')
plt.close()


# Histograms ####################################################################

# Data generation
data1 = np.random.normal(0, 1, 300)
data2 = np.random.normal(0, 0.5, 250)
data = np.concatenate((data1, data2))
# simple histogram
nbins = int((np.amax(data) - np.amin(data)) / 0.3)
plt.hist(data1, bins = nbins, histtype = 'step', color = 'b', label = 'dist 1')
plt.hist(data2, bins = nbins, histtype = 'step', color = 'r', label = 'dist 2')
plt.legend()
plt.xlabel('data')
plt.title('Two histograms')
plt.savefig(output_dir + 'Histograms.png')
plt.close()
# density histogram
plt.hist(data, bins = nbins, density = True)
plt.xlabel('data')
plt.title('Density histogram')
plt.savefig(output_dir + 'Density.png')
plt.close()
# cumulative
plt.hist(data, bins = nbins, cumulative = True)
plt.xlabel('data')
plt.title('Cumulative histogram')
plt.savefig(output_dir + 'Cumulative.png')
plt.close()
