# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:03:25 2019

@author: carstenschelp
"""
import numpy as np
import sys
sys.path.append('../src')
import autoboundary as abd
import normpca as npca


class MyDumbClusterer():

    def fit(self, X):
        return self._conquer(X)

    def _conquer(self, X):
        max_depth = 3
        labels = np.repeat(0, X.shape[0])
        depth = 0
        # TODO: remove depth-loop
        # _divide_and calculates and returns and takes cluster-quality
        # ... or introduce class var dict label => cluster_quality
        for depth in range(0, max_depth):
            for label in set(labels):
                ix_label = (labels == label)
                subX = X[ix_label]
                if subX.size <= X.shape[1]:
                    continue
                subLabels = self._divide_and(subX)
                labels[ix_label] *= 2
                labels[ix_label] += subLabels

        self.labels_ = labels

    def _divide_and(self, X):
        nrmpca = npca.NormPCA(X)
        deltas, sort_ix = abd._build_indexed_deltas(nrmpca.projection[:, 0])
        # TODO: create change of cluster quality as stop criterium.
        split_ixes = abd._split(deltas, sort_ix, rel_fade_size=0.3)
        return abd.ixes_to_labels(X.shape[0], split_ixes)
