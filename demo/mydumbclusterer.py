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

    def _is_progress(self, cq_parent, cq_child1, cq_child2):
        return cq_child1 > cq_parent and cq_child2 > cq_parent

    def _and_conquer(self, X):
        print()
        labels = np.repeat(0, X.shape[0])
        labels_to_process = {}
        # initial label zero with dummy cluster quality
        labels_to_process[self._get_next_label()] = 'dummy'
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

                subLabels = self._divide(subX, label)

                if not subLabels.any():
                    labels_to_process.pop(label)
                    continue

                new_labels = np.array([self._get_next_label(), self._get_next_label()])
                labels[ix_label] = new_labels[np.array(subLabels)]
                
                labels_to_process.pop(label)
                labels_to_process[new_labels[0]] = 'dummy'
                labels_to_process[new_labels[1]] = 'dummy'
                print(f'replaced label {label} with {new_labels[0]} and {new_labels[1]}')

#                ax = plt.subplot(111)
#                ax.scatter(X[:,0], X[:,1], color='grey', label='others')
#                
#                sub_low = subX[(subLabels == 0)]
#                ax.scatter(sub_low[:, 0], sub_low[:, 1], label=str(new_labels[0]))
#                
#                sub_high = subX[(subLabels == 1)]
#                ax.scatter(sub_high[:, 0], sub_high[:, 1], label=str(new_labels[1]))
#                ax.legend()
#                plt.show()

        self.labels_ = labels

    def _get_cluster_quality(self, deltas):
        variance = deltas.var()
        if variance == 0:
            print(f'Zero variance. deltas: {deltas}')
        return np.square(np.median(deltas)) / variance
    
    def _build_weight(self, n):
        if n < 3:
            return np.repeat(1, n)
        n_fade = max(int(n * self._rel_fade_size), 1)
        weights = np.repeat(n_fade, n)
        weights[:n_fade] = range(0, n_fade)
        weights[-n_fade:] = range(n_fade-1, -1, -1)
        return weights

    def _divide(self, X, label):
        nrmpca = npca.NormPCA(X)
        # switch to next eigenvector when split along
        # first eigenvector yields poorer cluster quality
        for pca_dimension in range(0, X.shape[1]):
            prj_values = nrmpca.projection[:, pca_dimension]
            deltas, sort_ix = \
                abd._build_indexed_deltas(prj_values)

            if self._rel_fade_size:
                weight = self._build_weight(len(deltas))
                weighted_deltas = deltas * weight
            else:
                weighted_deltas = deltas

            split_ixes = abd._split_by_deltas(weighted_deltas, sort_ix)
            range_total = prj_values[-1] - prj_values[0]
            range_low =  prj_values[split_ixes[0][-1]] - prj_values[split_ixes[0][0]]
            range_high =  prj_values[split_ixes[1][-1]] - prj_values[split_ixes[1][0]]
            cluster_quality = 1 - (range_low + range_high) / range_total
            # TODO: also evaluate relevance (subCount/totalCount)
            if cluster_quality > 0.25:
                print(f'Good cluster quality for label {label} at dimension {pca_dimension}.')
                return abd.ixes_to_labels(X.shape[0], split_ixes)

            print(f'stop at label {label} cluster quality: {cluster_quality}')
            return np.array(())

