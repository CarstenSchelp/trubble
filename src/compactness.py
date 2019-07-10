# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:19:55 2019

@author: carstenschelp
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

def compactness(a):
    uniform_range = a.max() - a.min()

    std = a.std()
    if std == 0:
        return np.inf
    normal_range =  6 * std
    return uniform_range / normal_range

normal = np.random.normal(size=1000)#.array([1,2.5,3.4,3.7,4.5,6])
uniform = np.random.uniform(size=1000)#np.array([2,4,6,8,10,12])
stretch = np.array([1,2,7,8,12,13])
hole = np.array([1,2,3,11,12,13])

plt.hist(normal)
plt.suptitle('normal')
print(f"compactness normal {compactness(normal)}")
plt.show()

plt.hist(uniform)
plt.suptitle('uniform')
print(f"compactness uniform {compactness(uniform)}")
plt.show()

plt.hist(stretch)
plt.suptitle('stretch')
print(f"compactness stretch {compactness(stretch)}")
plt.show()

plt.hist(hole)
plt.suptitle('hole')
print(f"compactness hole {compactness(hole)}")
plt.show()

