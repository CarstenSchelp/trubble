# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:28:52 2019

@author: carstenschelp
"""
#import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/')
import autoboundary as abd
sys.path.append('../test/')
import testsets as ts

#testsets = ts.create_normal_testsets()
testsets = ts.create_small_testsets()

fig, axes = plt.subplots(len(testsets), 4, figsize=( 4 * 6, len(testsets) * 2))

for row_ix in range(0, len(testsets)):
  testset = testsets[row_ix]
  name = testset['name']
  a = testset['values']
  axes[row_ix, 0].bar(range(0, len(a)), a, color='blue')
  axes[row_ix, 0].set_title(name)
  
  ix_lo, ix_hi = abd.argsplit(a)

  axes[row_ix, 1].bar(ix_lo, a[ix_lo], color='blue')
  axes[row_ix, 1].bar(ix_hi, a[ix_hi], color='red')
  axes[row_ix, 1].set_title('split')

  axes[row_ix, 2].bar(range(0, ix_lo.size), a[ix_lo], color='blue')
  axes[row_ix, 2].bar(
      range(ix_lo.size, ix_lo.size + ix_hi.size),
      a[ix_hi],
      color='red')
  axes[row_ix, 2].set_title('split and sorted')
  
  cluster_ixs = abd.argcluster(a)
  offset = 0
  for clust_ix in cluster_ixs:
    x = range(offset, offset + clust_ix.size)
    axes[row_ix, 3].bar(x, a[clust_ix])
    offset += clust_ix.size
  axes[row_ix, 3].set_title('clustered')
    
plt.subplots_adjust(hspace=0.7)
plt.show()
