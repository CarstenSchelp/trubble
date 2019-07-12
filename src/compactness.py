# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:19:55 2019

@author: carstenschelp
"""
# %%
import numpy as np
import matplotlib.pyplot as plt


def estimate_compactness(a):
    total_range = a.max() - a.min()
    std = a.std()
    if std == 0:
        return np.inf
    assumed_normal_range = 6 * std
    return total_range / assumed_normal_range


normal = np.random.normal(loc=100, scale=10, size=1000)
uniform = np.random.uniform(low=-10, high=10, size=1000)
linspace = np.linspace(-5, 7, 1000)
stretch = np.array([1, 2, 7, 8, 12, 13])
hole = np.array([1, 2, 3, 11, 12, 13])

plt.hist(normal)
plt.suptitle('normal')
print(f"estimate_compactness normal {estimate_compactness(normal)}")
plt.show()

plt.hist(linspace)
plt.suptitle('linspace')
print(f"estimate_compactness linspace {estimate_compactness(linspace)}")
plt.show()

plt.hist(uniform)
plt.suptitle('uniform')
print(f"estimate_compactness uniform {estimate_compactness(uniform)}")
plt.show()

plt.hist(stretch)
plt.suptitle('stretch')
print(f"estimate_compactness stretch {estimate_compactness(stretch)}")
plt.show()

plt.hist(hole)
plt.suptitle('hole')
print(f"estimate_compactness hole {estimate_compactness(hole)}")
plt.show()
