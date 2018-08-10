__author__ = 'wanqian'
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

centeroids = [[30,3, 3], [0,0,0], [-1,-1,-10], [-20,-2,-5]]
cluster_std = [0.2, 0.1, 0.2, 0.2]
dataset, label= make_blobs(n_samples=10000, n_features=3, centers=4, center_box=centeroids,
                           cluster_std=cluster_std, random_state =9)
print(dataset.shape, label.shape)

pca = PCA(n_components=2).fit(dataset)
data_new = pca.transform(dataset)

fig1 = plt.figure()
ax = fig1.add_subplot(1, 2, 1, projection='3d')
ax.scatter(dataset[:,0], dataset[:,1], dataset[:,2], c=label)
ax.set_title('inital data')

ax = fig1.add_subplot(1, 2, 2)
ax.scatter(data_new[:,0], data_new[:,1], c=label)
ax.set_title('pca data')
plt.show()