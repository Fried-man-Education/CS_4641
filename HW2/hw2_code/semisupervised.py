import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

def complete_(data): # [1pts]
    """
    Args:
        data: N x D numpy array where the last column is the labels 
    Return:
        labeled_complete: n x D array (n <= N) where values contain both complete features and labels
    """
    
    raise NotImplementedError
    
def incomplete_(data): # [1pts]
    """
    Args:
        data: N x D numpy array where the last column is the labels
    Return:
        labeled_incomplete: n x D array (n <= N) where values contain incomplete features but complete labels
    """    
    
    raise NotImplementedError

def unlabeled_(data): # [1pts]
    """
    Args:
        data: N x D numpy array where the last column is the labels   
    Return:
        unlabeled_complete: n x D array (n <= N) where values contain complete features but incomplete labels
    """
    
    raise NotImplementedError

class CleanData(object):
    def __init__(self): # No need to implement
        pass

    def pairwise_dist(self, x, y): # [0pts] - copy from kmeans
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        
        raise NotImplementedError
    
    def __call__(self, incomplete_points,  complete_points, K, **kwargs): # [7pts]
        """
        Function to clean or "fill in" NaN values in incomplete data points based on
        the average value for that feature for the K-nearest neighbors in the complete data points. 

        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points:   N_complete   x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_complete + N_incomplete) x (D+1) numpy array, containing both the complete points and recently filled points

        Notes: 
            (1) The first D columns are features, and the last column is the class label
            (2) There may be more than just 2 class labels in the data (e.g. labels could be 0,1,2 or 0,1,2,...,M)
            (3) There will be at most 1 missing feature value in each incomplete data point (e.g. no points will have more than one NaN value)
            (4) You want to find the k-nearest neighbors within each class separately;
            (5) There may be missing values in all of the features. It might be more convenient to address each feature at a time.
            (6) Do NOT use a for-loop over N_incomplete; you may use a for-loop over the M labels and the D features (e.g. omit one feature at a time) 
            (7) You do not need to order the rows of the return array clean_points in any specific manner
        """
        
        raise NotImplementedError


class SemiSupervised(object):
    def __init__(self): # No need to implement
        pass
    
    def softmax(self,logit): # [0 pts] - can use same as for GMM
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array where softmax has been applied row-wise to input logit
        """
        
        raise NotImplementedError

    def logsumexp(self,logit): # [0 pts] - can use same as for GMM
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:])
        """
        
        raise NotImplementedError
    
    def normalPDF(self, logit, mu_i, sigma_i): # [0 pts] - can use same as for GMM
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.diagonal() should be handy.
        """
        
        raise NotImplementedError
    
    def _init_components(self, points, K, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            
        Hint: (1) The paper describes how you should initialize your algorithm.
              (2) Use labeled points only
        """

        raise NotImplementedError

    def _ll_joint(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
        """
        
        raise NotImplementedError

    def _E_step(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """

        raise NotImplementedError

    def _M_step(self, points, gamma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
            
        Hint:  There are formulas in the slide.
        """

        raise NotImplementedError

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: N x (D+1) numpy array, where 
                - N is # points, 
                - D is the number of features,
                - the last column is the point labels (when available) or NaN for unlabeled points
            K: integer, number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            pi, mu, sigma: (1xK np array, KxD numpy array, KxDxD numpy array)
        """
        
        # initialize based only on labeled data
        pi, mu, sigma = None, None, None  # TODO
        pbar = tqdm(range(max_iters))
        for it in pbar:
            # E-step
            
            # M-step
            
            # calculate the negative log-likelihood of observation
            joint_ll = None  # TODO
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        
        raise NotImplementedError  


class ComparePerformance(object):

    def __init__(self): #No need to implement
        pass

    @staticmethod
    def accuracy_semi_supervised(training_data, validation_data, K:int) -> float: # [2.5 pts]
        """
        Train a classification model using your SemiSupervised object on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data 

        Args:
            training_data: N_t x (D+1) numpy array, where 
                - N_t is the number of data points in the training set, 
                - D is the number of features, and 
                - the last column represents the labels (when available) or a flag that allows you to separate the unlabeled data.
            validation_data: N_v x(D+1) numpy array, where 
                - N_v is the number of data points in the validation set,
                - D is the number of features, and 
                - the last column are the labels
            K: integer, number of clusters for SemiSupervised object
        Return:
            accuracy: floating number
        
        Note: (1) validation_data will NOT include any unlabeled points
              (2) you may use sklearn accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        """

        # Train SemiSupervised with training_data

        # Classify validation_data -- Hint: use _E_step; find max probability class for each data point

        # Score your model's classification of validation_data against the actual labels in validation_data
        
        raise NotImplementedError

    @staticmethod
    def accuracy_GNB(training_data, validation_data) -> float: # [2.5 pts]
        """
        Train a Gaussion Naive Bayes classification model (sklearn implementation) on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data 

        Args:
            training_data: N_t x (D+1) numpy array, where 
                - N is the number of data points in the training set, 
                - D is the number of features, and 
                - the last column represents the labels
            validation_data: N_v x (D+1) numpy array, where 
                - N_v is the number of data points in the validation set,
                - D is the number of features, and 
                - the last column are the labels
        Return:
            accuracy: floating number

        Note: (1) both training_data and validation_data will NOT include any unlabeled points
              (2) use sklearn implementation of Gaussion Naive Bayes: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        """

        # Train GaussianNB model with training_data

        # Classify validation_data 
        # Score GaussianNB model's classification of validation_data against the actual labels in validation_data
        # Hint: you can do both these steps with one line from sklearn
        
        raise NotImplementedError


