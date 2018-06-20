#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:18:25 2018

@author: xinjiwang
"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
data =pd.read_csv('raw_data.csv')
print(data.shape)
data.columns

X = list(zip(data['view'].values,data['click'].values,data['hover'].values))
kmeans = KMeans(n_clusters = 5)
kmeans = kmeans.fit(X)

labels = kmeans.predict(X)

centroids = kmeans.cluster_centers_
# Comparing with scikit-learn centroids

print(centroids) # From sci-kit learn

%pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data['view'].values, data['click'].values, data['hover'].values)
data['labels'] = labels
%pylab 
fig = plt.figure()
plt.xlabel("view")
plt.ylabel("load")
pylab.scatter(data['view'].values, data['load'].values, c=labels,cmap=pylab.cm.cool)

fig = plt.figure()
plt.xlabel("view")
plt.ylabel("click")
pylab.scatter(data['view'].values, data['click'].values, c=labels,cmap=pylab.cm.cool)
ax.scatter(data['view'].values, data['click'].values, data['hover'].values,c = labels)

fig = plt.figure()
plt.xlabel("view")
plt.ylabel("hover")
pylab.scatter(data['view'].values, data['hover'].values, c=labels,cmap=pylab.cm.cool)

