import numpy as np
from matplotlib import pyplot as plt

class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X): # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array (no need to transpose what is returned from np.linalg.svd)
        """
        X = (X - X.mean(axis=0))
        self.U, self.S, self.V = np.linalg.svd(
            X,
            full_matrices=False
        )

    def transform(self, data, K=2): # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        """
        return self.U[:, : K] * self.S[: K]

    def transform_rv(self, data, retained_variance=0.99): # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        """
        return self.transform(
            data,
            np.argwhere(
                np.cumsum(
                    np.square(self.S)[
                        np.argsort(
                            np.square(self.S)
                        )[:: -1]
                    ]
                ) / np.sum(
                    np.square(self.S)
                ) >= retained_variance
            )[0][0] + 1
        )

    def get_V(self):
        """ Getter function for value of V """
        
        return self.V


    def visualize(self, X, y, fig=None):   # 5 pts
        """
        Use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color.
        Hint: To create the scatter plot, it might be easier to loop through the labels (Plot all points in class '0', and then class '1')
        Hint: To reproduce the scatter plot in the expected outputs, use the colors 'blue', 'magenta', and 'red' for classes '0', '1', '2' respectively.
        
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels
            
        Return: None
        """
        self.fit(X)
        plt.scatter(self.transform(X, K=2)[y == 0, 0], self.transform(X, K=2)[y == 0, 1], c='blue', label='0')
        plt.scatter(self.transform(X, K=2)[y == 1, 0], self.transform(X, K=2)[y == 1, 1], c='magenta', label='1')
        plt.scatter(self.transform(X, K=2)[y == 2, 0], self.transform(X, K=2)[y == 2, 1], c='red', label='2')

        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend()
        plt.show()
