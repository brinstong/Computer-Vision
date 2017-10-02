import pylab
from matplotlib import pyplot as plt
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
import cv2
import numpy as np

NUM_CLUST = 6

distSqMat = np.loadtxt('/home/brinstongonsalves/Documents/PyCharm/CV/mat.txt')
link_mat = hier.linkage(distSqMat,'single')
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram : Full')
hier.dendrogram(link_mat)
plt.savefig("dendogram.jpg")

plt.clf()
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram : Truncated')
hier.dendrogram(link_mat,truncate_mode='lastp',p = NUM_CLUST,)
plt.savefig("dendogram1.jpg")
