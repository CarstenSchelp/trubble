import numpy as np
# import pandas as pd


class NormPCA:
    def __init__(self, data):
        if data.shape[0] < data.shape[1]:
            raise ValueError(f'"data" should have more rows than columns.')
        order = data.shape[1]
        self.mean = data.mean(axis=0)
        self.covariance = np.cov(data, rowvar=False, bias=True)
        self.variance = self.covariance.diagonal()
        var = np.repeat(self.variance, order)
        var.shape = order, order
        self.corrcoef = self.covariance / np.sqrt(var * var.T)

        eigval, eignvect = np.linalg.eig(self.corrcoef)
        sort_ix = np.argsort(eigval)
        self.eigenvalues = eigval[sort_ix]
        self.eigenvectors = eignvect[sort_ix]
        self.projection = data.dot(self.eigenvectors.T)

# TODO: help with dim-reduction decisions.
# Big/small eigenvalues, preserved variance score ...
