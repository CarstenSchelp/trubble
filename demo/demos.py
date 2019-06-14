# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:28:52 2019

@author: carstenschelp
"""
# import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../src/')
sys.path.append('../test/')
import autoboundary as abd
import testsets as ts


# testsets = ts.create_normal_testsets()
testsets = ts.create_small_testsets()

# demo splitting

n_columns = 2
fig, axes = plt.subplots(
    len(testsets),
    n_columns,
    figsize=(n_columns * 6, len(testsets) * 2))

for row_ix in range(0, len(testsets)):
    testset = testsets[row_ix]
    name = testset['name']
    a = testset['values']

    ix_lo, ix_hi = abd.argsplit(a)

    axes[row_ix, 0].bar(ix_lo, a[ix_lo], color='blue')
    axes[row_ix, 0].bar(ix_hi, a[ix_hi], color='red')
    axes[row_ix, 0].set_title(f'{name} (split)')

    axes[row_ix, 1].bar(range(0, ix_lo.size), a[ix_lo], color='blue')
    axes[row_ix, 1].bar(
        range(ix_lo.size, ix_lo.size + ix_hi.size),
        a[ix_hi],
        color='red')
    axes[row_ix, 1].set_title(f'{name} (sorted and split)')

    #print('split labels:', abd.labelsplit(a))

plt.subplots_adjust(hspace=0.7)
plt.show()

# demo clustering
n_columns = 2
fig, axes = plt.subplots(
    len(testsets),
    n_columns,
    figsize=(n_columns * 6, len(testsets) * 2))

for row_ix in range(0, len(testsets)):
    testset = testsets[row_ix]
    name = testset['name']
    a = testset['values']
    cluster_ixs = abd.argcluster(a)
    offset = 0
    for clust_ix in cluster_ixs:
        axes[row_ix, 0].bar(clust_ix, a[clust_ix])
        offset += clust_ix.size
    axes[row_ix, 0].set_title(f'{name} (clustered)')

    offset = 0
    for clust_ix in cluster_ixs:
        x = range(offset, offset + clust_ix.size)
        axes[row_ix, 1].bar(x, a[clust_ix])
        offset += clust_ix.size
    axes[row_ix, 1].set_title(f'{name} (sorted and clustered)')

    #print('cluster labels:', abd.labelcluster(a))

plt.subplots_adjust(hspace=0.7)
plt.show()
