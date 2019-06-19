# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:03:25 2019

@author: carstenschelp
"""
import numpy as np
import sys

# debug
import matplotlib.pyplot as plt

sys.path.append('../src')
import autoboundary as abd
import normpca as npca


class MyDumbClusterer():
    
    def __init__(self):
        self._rel_fade_size = 0.5
        self._next_label = 0
    
    def fit(self, X):
        return self._and_conquer(X)
    
    def _get_next_label(self):
        next_label = self._next_label
        self._next_label += 1
        return next_label

    def _and_conquer(self, X):
        print()
        labels = np.repeat(0, X.shape[0])
        labels_to_process = {}
        # initial label zero with dummy cluster quality
        labels_to_process[self._get_next_label()] = 0
        # TODO: plot after every split!
        while labels_to_process:
            for label in list(labels_to_process.keys()):
                print('to process:', labels_to_process.keys())
                ix_label = (labels == label)

                subX = X[ix_label]
                if len(subX) < X.shape[1]:
                    # cannot apply pca when there are more
                    # dimensions than observations (more columns than rows)
                    labels_to_process.pop(label)
                    continue

                subLabels, cq_parent, cq_low, cq_high = \
                    self._divide(subX, labels_to_process[label], label)
                if not labels_to_process[label]:
                    # add calculated cluster quality for first label (zero)
                    labels_to_process[label] = cq_parent

                if cq_low < cq_parent or cq_high < cq_parent:
                    print(f'stop at label {label} cqs: {cq_parent, cq_low, cq_high}')
                    labels_to_process.pop(label)
                    continue

                new_labels = np.array([self._get_next_label(), self._get_next_label()])
                labels[ix_label] = new_labels[np.array(subLabels)]
                
                labels_to_process.pop(label)
                labels_to_process[new_labels[0]] = cq_low
                labels_to_process[new_labels[1]] = cq_high
                print(f'replaced label {label} with {new_labels[0]} and {new_labels[1]}')
                ax = plt.subplot(111)               
                ax.scatter(X[:, 0], X[:, 1], s=10, c=labels)
                plt.show()

        self.labels_ = labels

    def _get_cluster_quality(self, deltas):
        variance = deltas.var()
        if variance == 0:
            print(f'Zero variance. deltas: {deltas}')
        return np.square(deltas.mean()) / variance
    
    def _build_weight(self, n):
        if n < 3:
            return np.repeat(1, n)
        n_fade = max(int(n * self._rel_fade_size), 1)
        weights = np.repeat(n_fade, n)
        weights[:n_fade] = range(0, n_fade)
        weights[-n_fade:] = range(n_fade-1, -1, -1)
        return weights

    def _divide(self, X, cq_parent, label):
        nrmpca = npca.NormPCA(X)
        # switch to next eigenvector when split along
        # first eigenvector yields poorer cluster quality
        for pca_dimension in range(0, X.shape[1]):
            deltas, sort_ix = \
                abd._build_indexed_deltas(nrmpca.projection[:, pca_dimension])

            if not cq_parent:
                cq_parent = self._get_cluster_quality(deltas)

            if self._rel_fade_size:
                weight = self._build_weight(len(deltas))
                deltas *= weight

            split_ixes = abd._split_by_deltas(deltas, sort_ix)
            cq_low = self._get_cluster_quality(deltas[split_ixes[0]])
            cq_high = self._get_cluster_quality(deltas[split_ixes[1]])
            if cq_low >= cq_parent and cq_high >= cq_parent:
                print(f'Good sub cluster qualities for label {label} at dimension {pca_dimension}.')
                break

        return abd.ixes_to_labels(X.shape[0], split_ixes), \
            cq_parent, cq_low, cq_high
