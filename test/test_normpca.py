#!/usr/bin/python3
'''
Tests the online_cov module
'''
import numpy as np
import sys
sys.path.append('../src/')
from normpca import NormPCA
sys.path.append('../test/')
import testsets as ts


plot_results = False

# %%

def test_project_01():
    data = np.array([
            [1, 8],
            [3, 6],
            [5, 4],
            [7, 2]])
    nrm_pca = NormPCA(data)

    assert (nrm_pca.mean == [4, 5]).all(),\
        f"""mean should be [4, 5].
        Was\n {nrm_pca.mean}."""

    assert (nrm_pca.variance == [5, 5]).all(),\
        f"""variance should be [5, 5]. Was\n {nrm_pca.variance}."""

    assert (nrm_pca.corrcoef == [[1, -1], [-1, 1]]).all(), \
        f"""corrcoeffs should indicate total negative correlation (-1)
        Was\n{nrm_pca.corrcoef}."""

    assert nrm_pca.projection.shape == data.shape, \
        f"""projection should have same shape as original data.
        Was\n{nrm_pca.projection.shape}."""

    assert np.isclose(np.var(nrm_pca.projection[:, 1]), 0),\
        f"""Total correlation should result in second PCA-axis
        having zero variance.Was\n{nrm_pca.projection[:, 1]}"""

    if plot_results:
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("original data")
        plt.show()

        plt.scatter(nrm_pca.projection[:, 0], nrm_pca.projection[:, 1])
        plt.title("projected data")
        plt.show()


def test_project_02():
    data = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
    nrm_pca = NormPCA(data)

    assert (nrm_pca.mean == [4, 5]).all(),\
        f"""mean should be [4, 5].
        Was\n {nrm_pca.mean}."""

    assert (nrm_pca.variance == [5, 5]).all(),\
        f"""variance should be [5, 5]. Was\n {nrm_pca.variance}."""

    assert (nrm_pca.corrcoef == [[1, 1], [1, 1]]).all(), \
        f"""corrcoeffs should indicate total positive correlation (+1)
        Was\n{nrm_pca.corrcoef}."""

    assert nrm_pca.projection.shape == data.shape, \
        f"""projection should have same shape as original data.
        Was\n{nrm_pca.projection.shape}."""

    assert np.isclose(np.var(nrm_pca.projection[:, 1]), 0),\
        f"""Total correlation should result in second PCA-axis
        having zero variance.Was\n{nrm_pca.projection[:, 1]}"""

    if plot_results:
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("original data")
        plt.show()

        plt.scatter(nrm_pca.projection[:, 0], nrm_pca.projection[:, 1])
        plt.title("projected data")
        plt.show()


def test_project_03():
    data = np.array([
            [-1, -1],
            [4, 4],
            [4, -1],
            [-1, 4]])
    nrm_pca = NormPCA(data)

    assert (nrm_pca.mean == [1.5, 1.5]).all(),\
        f"""mean should be [1.5, 1.5].
        Was\n {nrm_pca.mean}."""

    assert nrm_pca.variance[0] == nrm_pca.variance[1],\
        f"""variances should be the same. Was\n {nrm_pca.variance}."""

    assert (nrm_pca.corrcoef == [[1, 0], [0, 1]]).all(), \
        f"""corrcoeffs should indicate no correlation (zero)
        Was\n{nrm_pca.corrcoef}."""

    assert (data == nrm_pca.projection).all(), \
        f"""With no correlation at all, projected data should
        be the same as original data.
        Was\n{nrm_pca.corrcoef}."""

    if plot_results:
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("original data")
        plt.show()

        plt.scatter(nrm_pca.projection[:, 0], nrm_pca.projection[:, 1])
        plt.title("projected data")
        plt.show()


def test_plot_projection_00():
    if not plot_results:
        return
    datasets = ts.create_clustering_testsets()
    fig, axes = plt.subplots(nrows=2, ncols=len(datasets),
                             figsize=(3 * len(datasets),2 * 3))
    for ix, data in enumerate(datasets):
        X, labels = data
        npca = NormPCA(X)
        x_prj = npca.projection
        axes[0, ix].scatter(X[:, 0], X[:, 1], s=0.5)
        axes[1, ix].scatter(x_prj[:, 0], x_prj[:, 1], s=0.5)
    plt.show()


if __name__ == "__main__":
    plot_results = True
    import matplotlib.pyplot as plt
    test_project_01()
    test_project_02()
    test_project_03()
    test_plot_projection_00()
    print("ALL PASSED!")
