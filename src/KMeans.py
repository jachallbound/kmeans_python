import numpy as np
import numpy.random as npr
import ipdb

class KMeans:
    """Class for performing KMeans classification.
    Find K integer means in a cluster of data that optimizes the decision
    boundaries.
    Choose any integer of K means to find.
    Choose any dimension of cluster data to operate on.
    """
    def __init__(self, K, data):
        """Initialize Class

        Parameters:
        ---
        K : *int*
            Amount of means you wish to find in the data.
        data : *array_like*
            Cluster of data. Any amount of dimensions.
            data.shape[0] must be the amount of dimensions
        """
        self.K = K
        self.data = data
        self.ndim = data.shape[0]

    def get_means(self, random=False):
        """Method to calculate and return new K means

        """
        if random:
            # Get random means
            K_mat = np.zeros([self.K, self.ndim])
            for k in range(self.K):
                for d in range(self.ndim):
                    K_mat[k,d] = npr.uniform(
                        np.min(self.data[d,]),np.max(self.data[d,])
                        )
        else:
            # Get mean from data
            K_mat = 1
        return K_mat
