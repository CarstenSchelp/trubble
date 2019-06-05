# -*- coding: utf-8 -*-
"""
Created on Tue Jun    4 21:25:50 2019

@author: carstenschelp
"""
import numpy as np


def create_small_testsets():
    testsets = []
    np.random.seed(3)

    testsets.append({
        'name': 'random',
        'values': np.random.uniform(0.0, 100, 12) + 1})

    testsets.append({
        'name': 'one obvious leap',
        'values': np.array((3, 8, 3, 8, 3, 8, 8, 3, 3))})

    two_obvious_leaps = np.array([2, 3, 4, 9, 10, 11, 15, 16, 17])
    np.random.shuffle(two_obvious_leaps)
    testsets.append({
        'name': 'two obvious leaps',
        'values': two_obvious_leaps})

    testsets.append({
        'name': 'all the same',
        'values': np.array((4, 4, 4, 4, 4, 4, 4, 4, 4))})

    testsets.append({
        'name': 'low outlier',
        'values': np.array((9, 9, 3, 9, 9, 9, 9, 9, 9))})

    testsets.append({
        'name': 'high outlier',
        'values': np.array((5, 5, 5, 5, 5, 5, 10, 5, 5))})

    incremental = np.array((2, 4, 6, 8, 10, 12, 14))
    np.random.shuffle(incremental)
    testsets.append({
        'name': 'exact same deltas',
        'values': incremental})

    noisy_incremental = np.array((2, 4, 6, 8, 10, 12, 14)) \
        * np.random.uniform(low=0.90, high=1.1, size=7)
    np.random.shuffle(noisy_incremental)
    testsets.append({
        'name': 'similar deltas with noise',
        'values': noisy_incremental})

    exponential = np.exp(np.arange(1, 4, 0.2))
    np.random.shuffle(exponential)
    testsets.append({
        'name': 'exponential',
        'values': exponential})

    testsets.append({
        'name': 'just two elements',
        'values': np.array((3, 1))})

    testsets.append({
        'name': 'just one element',
        'values': np.array((3,))})

    testsets.append({
        'name': 'empty',
        'values': np.empty((0, 0))})

    return testsets


def create_normal_testsets():
    testsets = []
    np.random.seed(3)

    testsets.append({
        'name': 'single normal dist',
        'values': np.random.normal(100.0, 20, 30)})

    norm1 = np.random.normal(100.0, 15, 20)
    norm2 = np.random.normal(60, 20, 20)
    both = np.concatenate((norm1, norm2))
    np.random.shuffle(both)
    testsets.append({
        'name': 'two normal dists',
        'values': both})

    return testsets
