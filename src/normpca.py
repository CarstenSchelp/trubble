import numpy as np
# import pandas as pd


class NormPCA:
    def __init__(self, data):
        if data.shape[0] < data.shape[1]:
            raise ValueError(f'"data" should have more rows than columns. ({data.shape[0]} rows, {data.shape[1]} columns)')
        self.order = data.shape[1]
        self.data = data
        self.mean = data.mean(axis=0)
        self.covariance = np.cov(data, rowvar=False, bias=True)
        self.variance = self.covariance.diagonal()
        var = np.repeat(self.variance, self.order)
        var.shape = self.order, self.order
        self.corrcoef = self.covariance / np.sqrt(var * var.T)

        # TODO: do sorting in projection
        eigval, eigvect = np.linalg.eig(self.corrcoef)
        sort_ix = np.flip(np.argsort(eigval))
        
        self.eigenvalues = eigval[sort_ix]
        self.eigenvectors = eigvect[sort_ix]
        
    def get_projection(self):
        projection = self.data.dot(self.eigenvectors.T)
        # TODO: options: sort by ascending compactness (descending raggedness)
        # ... or by descending eigenvalue (=variance)
        # ... or specify minimum variance covered
        # ... or split split eigenvalues at significant drop
        return projection
        
    def estimate_compactness(a):
        total_range = a.max() - a.min()
        std = a.std()
        if std == 0:
            return np.inf
        assumed_normal_range = 6 * std
        return total_range / assumed_normal_range
                
        
#        self.covered_variance = \
#            np.cumsum(self.eigenvalues) / self.eigenvalues.sum()

# TODO: help with dim-reduction decisions.
# Big/small eigenvalues, preserved variance score ...
