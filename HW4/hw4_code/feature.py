import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    output = np.zeros((
        X.shape[0],
        1
    ))
    for i in range(X.shape[0]): output[i] = X[i][0] * X[i][1]
    return np.append(X, output, axis=1)
