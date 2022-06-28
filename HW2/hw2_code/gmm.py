import numpy as np
from tqdm import tqdm
from kmeans import KMeans

SIGMA_CONST = 1e-6  # Only add SIGMA_CONST when sigma_i is not invertible
LOG_CONST = 1e-32

FULL_MATRIX = False  # Set False if the covariance matrix is a diagonal matrix


class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """

        temp = np.atleast_2d(logit)
        temp -= np.expand_dims(
            np.max(temp, axis=1),
            axis=1
        )
        temp = np.exp(temp)

        sig = np.expand_dims(
            np.sum(temp, axis=1),
            axis=1
        )

        return temp / sig

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        temp = list(np.asarray(logit).shape)
        temp[1] = 1
        top = np.asarray(logit).max(axis=1)
        total = np.log(
            np.exp(
                np.asarray(logit) - top.reshape(temp)
            ).sum(axis=1)
        )
        output = (top + total).reshape(temp[0], 1)
        return output

    # for undergraduate student
    def normalPDF(self, logit, mu_i, sigma_i):  # [5pts]
        """
        Args:
            logit: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        temp = np.exp(
            -0.5 * (logit - mu_i) ** 2 / sigma_i.diagonal()
        ) / np.sqrt(
            2 * np.pi * sigma_i.diagonal()
        )
        return np.prod(temp, axis=1)

    # for grad students
    def multinormalPDF(self, logits, mu_i, sigma_i):  # [5pts]
        """
        Args:
            logit: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. The value in self.D may be outdated and not correspond to the current dataset,
            try using another method involving the current arguments to get the value of D
        """

        raise NotImplementedError

    def _init_components(self, **kwargs):  # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        """
        pi = np.ones(self.K) * (1.0 / self.K)
        mu = self.points[
            np.random.choice(
                self.N,
                size=self.K,
                replace=False
            )
        ]
        sigma = np.array(
            [np.eye(self.points.shape[1]) for _ in range(self.K)]
        )
        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        output = np.empty((len(self.points), len(mu)))
        for x in range(len(mu)):
            output[:, x] = np.log(pi[x] + 1e-32) + np.log(self.normalPDF(self.points, mu[x], sigma[x]) + 1e-32)
        return output

    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        return self.softmax(
            self._ll_joint(pi, mu, sigma)
        )

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
        """
        temp = np.sum(
            gamma,
            axis=0
        )

        mu = np.array(
            [
                np.sum(
                    gamma[:, x].reshape(
                        self.N,
                        1
                    ) * self.points,
                    axis=0
                ) / temp[x] for x in range(self.K)
            ]
        )

        sigma = np.array(
            [
                np.matmul(
                    (gamma[:, x].reshape(self.N, 1) * (self.points - mu[x].reshape(
                        1,
                        self.D
                    ))).T,
                    self.points - mu[x].reshape(
                        1,
                        self.D
                    )
                ) / temp[x] for x in range(self.K)
            ]
        )

        pi = temp / self.N

        return pi, mu, sigma

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)
