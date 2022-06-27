'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def _init_centers(self, points, K, **kwargs):  # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        points = np.unique(
            points,
            axis=0
        )
        return points[
            np.random.permutation(points.shape[0])
            [: min(K, points.shape[0])]
        ]

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        cluster_idx = np.transpose(
            np.argmin(
                pairwise_dist(points, centers),
                axis=1
            )
        )
        return cluster_idx

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, a new K x D numpy array, where K is the number of clusters, and D is the dimension.

        HINT: If you need to reduce the number of clusters when there are 0 points for a center, then do so.
        """
        output = np.copy(old_centers)
        counter = 0
        for item in range(old_centers.shape[0]):
            curr = np.where(cluster_idx == item)[:1]
            output[counter, :] = np.mean(
                points[curr],
                axis=0
            )
            counter += 1
        return output

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """

        return (
                np.linalg.norm(
                    points - centers[cluster_idx]
                )
                ** 2
        ).sum()

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss


def find_optimal_num_clusters(image, max_K=15):  # [10 pts]
    np.random.seed(1)
    """Plots loss values for different number of clusters in K-Means

    Args:
        image: input image of shape(H * W, 3)
        max_K: number of clusters
    Return:
        losses: vector of loss values (also plot loss values against number of clusters but do not return this)
    """
    handle = image.shape[1]
    accumulation = []
    for i in range(1, max_K + 1):
        temp = KMeans()(np.reshape(image, [-1, handle]).astype(np.float32), i)[2]
        accumulation.append(temp)
    plt.plot(np.arange(1, max_K + 1), accumulation)
    return accumulation


def pairwise_dist(x, y):  # [5 pts]
    np.random.seed(1)
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
        dist: N x M array, where dist2[i, j] is the euclidean distance between
        x[i, :] and y[j, :]
    """
    dist = np.sqrt(
        np.sum(
            np.square(x)[:, np.newaxis, :], axis=2
        ) - 2
        * x.dot(y.T)
        + np.sum(
            np.square(y),
            axis=1
        )
    )
    return dist
