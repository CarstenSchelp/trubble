# -*- coding: utf-8 -*-
"""
Created on Mon Apr    8 12:32:30 2019

@author: carstenschelp
"""
import numpy as np


def _validate_one_dimensional(a):
    if len(a.shape) != 1:
        raise ValueError('Input must be a one-dimensional array.')


def _build_indexed_deltas(a):
    sort_ix = a.argsort()
    # build deltas
    deltas = np.diff(a[sort_ix])
    # prepend zero in order to make length of
    # deltas match length of original values
    deltas = np.insert(deltas, 0, 0)
    return deltas, sort_ix


def _deltas_are_homogenous(deltas):
    return len(deltas) > 2 and (deltas[1:-2] == deltas[-1]).all()


def _build_weight(n, rel_fade_size):
    if n < 3:
        return np.repeat(1, n)
    n_fade = max(int(n * rel_fade_size), 1)
    weights = np.repeat(n_fade, n)
    weights[:n_fade] = range(0, n_fade)
    weights[-n_fade:] = range(n_fade-1, -1, -1)
    return weights


def _split(deltas, sort_ix, rel_fade_size=0.0):
    if _deltas_are_homogenous(deltas):
        return np.empty((0, 0)).astype(int), sort_ix

    if rel_fade_size:
        weight = _build_weight(len(deltas), rel_fade_size=rel_fade_size)
        deltas *= weight

    # Find the index of the highest delta value.
    ix_max = deltas.argmax()
    ix_lo_a = sort_ix[0:ix_max]
    ix_hi_a = sort_ix[ix_max:]
    return ix_lo_a, ix_hi_a


def argsplit(a, rel_fade_size=0.0):
    if len(a) == 0:
        return np.empty((0, 0)).astype(int), np.empty((0, 0)).astype(int)

    _validate_one_dimensional(a)
    deltas, sort_ix = _build_indexed_deltas(np.array(a))

    return _split(deltas, sort_ix, rel_fade_size=0.0)


def ixes_to_labels(n, ixes):
    labels = np.empty(n, dtype=int)
    for label, ix in enumerate(ixes):
        np.put(labels, ix, label)
    return labels


def labelsplit(a, rel_fade_size=0.0):
    ixes = argsplit(a, rel_fade_size=rel_fade_size)
    return ixes_to_labels(a.shape, ixes)


def _get_boundary_ix(ix_high_deltas, highest_index, max_clusters=None):
    # Add lowest and highest overall index to the indices.
    # In order to get also the lowest and highest cluster complete.
    ix_cluster_boundaries = set(ix_high_deltas)
    ix_cluster_boundaries.add(0)
    ix_cluster_boundaries.add(highest_index)
    # sort on index so that we can apply ranges on it.
    boundaries = np.sort(list(ix_cluster_boundaries))

    if max_clusters:
        max_boundaries = max_clusters + 1
        if boundaries.size > max_boundaries:
            # hold on to greatest discarded delta value
            greatest_discarded_delta = boundaries[-max_boundaries]
            # truncate leading indices (those of the lower deltas)
            boundaries = boundaries[-max_clusters:]
            # to be consistent, discard possibly remaining
            # deltas that are equal to greatest discarded delta valye
            boundaries = boundaries[boundaries > greatest_discarded_delta]
            # re-prepend the very first index (zero).
            boundaries = np.insert(boundaries, 0, 0)

    return zip(boundaries[:-1], boundaries[1:])


def argcluster(a, max_clusters=None):
    if max_clusters and max_clusters <= 1:
        raise ValueError('kwarg "max_clusters" must be an integer > 1.')
    if len(a) == 0:
        return ()

    _validate_one_dimensional(a)
    deltas, sort_ix = _build_indexed_deltas(np.array(a))

    ix_low_deltas, ix_high_deltas = argsplit(np.log(1 + deltas))

    # When each element gets a cluster of its own
    # there is no point in clustering.
    # Return one cluster with all elements.
    if not ix_low_deltas.any():
        return sort_ix,

    ix_cluster_boundaries = _get_boundary_ix(
            ix_high_deltas,
            len(a),
            max_clusters)

    clusters = []
    for lo, hi in ix_cluster_boundaries:
        clusters.append(sort_ix[lo:hi])

    return tuple(clusters)


def labelcluster(a, max_clusters=None):
    cluster_ixes = argcluster(a, max_clusters=max_clusters)
    return ixes_to_labels(a.shape, cluster_ixes)
