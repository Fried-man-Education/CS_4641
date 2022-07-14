import numpy as np
RANDOM_SEED = 5


class NaiveBayes(object):
    def __init__(self):
        pass

    def likelihood_ratio(self, ratings_Sentiments):  # [5pts]
        '''
        Args:
            ratings_Sentiments: a python list of numpy arrays that is length <number of labels> x 1
            
            Example rating_Sentiments for three-label NB model:
    
            ratings_Sentiments = [ratings_1, ratings_2, ratings_3] -- length 3
            ratings_1: N_ratings_1 x D
                where N_ratings_1 is the number of negative news that we have,
                and D is the number of features (we use the word count as the feature)
            ratings_2: N_ratings_2 x D
                where N_ratings_2 is the number of neutral news that we have,
                and D is the number of features (we use the word count as the feature)
            ratings_3: N_ratings_3 x D
                where N_ratings_3 is the number of positive news that we have,
                and D is the number of features (we use the word count as the feature)
            
        Return:
            likelihood_ratio: (<number of labels>, D) numpy array, the likelihood ratio of different words for the different classes of sentiments.
        '''
        raise NotImplementedError

    def priors_prob(self, ratings_Sentiments):  # [5pts]
        '''
        Args:
            ratings_Sentiments: a python list of numpy arrays that is length <number of labels> x 1
            
            Example rating_Sentiments for Three-label NB model:
    
            ratings_Sentiments = [ratings_1, ratings_2, ratings_3] -- length 3
            ratings_1: N_ratings_1 x D
                where N_ratings_1 is the number of negative news that we have,
                and D is the number of features (we use the word count as the feature)
            ratings_2: N_ratings_2 x D
                where N_ratings_2 is the number of neutral news that we have,
                D is the number of features (we use the word count as the feature)
            ratings_3: N_ratings_3 x D
                where N_ratings_3 is the number of positive news that we have,
                and D is the number of features (we use the word count as the feature)
            
        Return:
            priors_prob: (1, <number of labels>) numpy array, where each entry denotes the prior probability for each class
        '''
        raise NotImplementedError

    
    def analyze_sentiment(self, likelihood_ratio, priors_prob, X_test):  # [5pts]
        '''
        Args:
            likelihood_ratio: (<number of labels>, D) numpy array, the likelihood ratio of different words for different classes of ratings
            priors_prob: (1, <number of labels>) numpy array, where each entry denotes the prior probability for each class
            X_test: (N_test, D) numpy array, a bag of words representation of the N_test number of ratings that we need to analyze
        Return:
            ratings: (N_test,) numpy array, where each entry is a class label specific for the Naïve Bayes model
        '''
        raise NotImplementedError


