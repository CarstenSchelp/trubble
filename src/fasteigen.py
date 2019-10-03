# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:05:27 2019

@author: carstenschelp
"""

# FAST EIGEN

import numpy as np
#%%
def get_correlated_dataset(k, n, dependency, mu, scale):
    latent = np.random.randn(n, k)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return np.array(scaled_with_offset)
#%%

np.random.seed(0)
n = 3
dependency = np.random.randn(n, n)
mu = np.zeros(n)
scale= np.ones(n)
data = get_correlated_dataset(n, 100, dependency, mu, scale)

#cov = np.cov(data, rowvar=False)
pearson = np.corrcoef(data, rowvar=False)

#print("cov on scale:", cov)
#print("numpy eigen on scale:", np.linalg.eig(cov))

print("cov normalized:", pearson)
print("numpy eigen normalized:", np.linalg.eig(pearson))

#%%
sqrt2 = np.sqrt(2)

# x values at which the other dimension is 1
x_tangentpoint_aligned_ellipsoid = (1 + pearson)/sqrt2
y_tangentpoint_aligned_ellipsoid = (1 - pearson)/sqrt2

x_tangent_to_aligned_ellipsoid = 1/sqrt2
y_tangent_to_aligned_ellipsoid = -1/sqrt2

y_vertical_tangent = np.sqrt(
        np.square(
            x_tangent_to_aligned_ellipsoid
            -
            x_tangentpoint_aligned_ellipsoid)
        +
        np.square(
            y_tangent_to_aligned_ellipsoid
            -
            y_tangentpoint_aligned_ellipsoid)
        )

points_squared = np.square(y_vertical_tangent)
print('points squared', points_squared)

one_by_r_square = np.linalg.solve(points_squared, np.ones(n))
print('one_by_r_square:', one_by_r_square)
egvals = 1 / np.sqrt(one_by_r_square)
print(egvals)
