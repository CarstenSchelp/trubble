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
    a_sorted = a[sort_ix]
    # build deltas
    deltas = np.diff(a_sorted)
    # prepend zero in order to make length of
    # deltas match length of original values
    deltas = np.insert(deltas, 0, 0)
    return a_sorted, deltas, sort_ix


def deltas_are_homogenous(deltas):
    return len(deltas) > 2 and (deltas[1:-2] == deltas[-1]).all()


def argsplit(a):
    if len(a) == 0:
        return np.empty((0, 0)).astype(int), np.empty((0, 0)).astype(int)

    _validate_one_dimensional(a)
    a_sorted, deltas, sort_ix = _build_indexed_deltas(np.array(a))

    if deltas_are_homogenous(deltas):
        return np.empty((0, 0)).astype(int), sort_ix

    # Find the index of the highest delta value.
    ix_max = deltas.argmax()
    ix_lo_a = sort_ix[0:ix_max]
    ix_hi_a = sort_ix[ix_max:]
    return ix_lo_a, ix_hi_a


def _get_boundary_ix(ix_high_deltas, highest_index, max_clusters=None):
    # Add lowest and highest overall index to the indices.
    # In order to get also the lowest and highest cluster complete.
    ix_cluster_boundaries = set(ix_high_deltas)
    ix_cluster_boundaries.add(0)
    ix_cluster_boundaries.add(highest_index)
    # sort on index so that we can apply ranges on it.
    flat_boundaries = np.sort(list(ix_cluster_boundaries))

    if max_clusters:
        max_boundaries = max_clusters + 1
        if flat_boundaries.size > max_boundaries:
            # truncate leading indices (those of the lower deltas)
            low_delta = flat_boundaries[-max_boundaries]
            flat_boundaries = flat_boundaries[-max_clusters:]
            flat_boundaries = flat_boundaries[flat_boundaries > low_delta]
            # re-prepend the very first index (zero).
            flat_boundaries = np.insert(flat_boundaries, 0, 0)

    return zip(flat_boundaries[:-1], flat_boundaries[1:])


def argcluster(a, max_clusters=None):
    if max_clusters and max_clusters <= 1:
        raise ValueError('kwarg "max_clusters" must be an integer > 1.')
    if len(a) == 0:
        return ()

    _validate_one_dimensional(a)
    a_sorted, deltas, sort_ix = _build_indexed_deltas(np.array(a))

    ix_low_deltas, ix_high_deltas = argsplit(deltas)

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
