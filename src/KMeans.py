import numpy as np
import numpy.random as npr
import ipdb

class KMeans():
    """Class for performing KMeans classification.
    Find K integer means in a cluster of data that optimizes the decision
    boundaries.
    Choose any integer of K means to find.
    Choose any dimension of cluster data to operate on.
    """
    def __init__(self, K, data, labels):
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
        self.labels = labels
        self.labels_p = np.zeros_like(labels)

    def generate_means(self, random=False):
        """Method to calculate and return new K means

        """
        K_mat = np.zeros([self.K, self.ndim])
        for k in range(self.K):
            for d in range(self.ndim):
                if random: # Get random means
                    K_mat[k,d] = npr.uniform(np.min(self.data[d, ]), np.max(self.data[d, ]))
                else: # Get means from mapped points
                    #ipdb.set_trace()
                    K_mat[k,d] = np.mean(self.data[d, np.where(self.labels_p == k)], axis=1)
                    if np.isnan(K_mat[k,d]):
                        K_mat[k,d] = npr.uniform(np.min(self.data[d, ]), np.max(self.data[d, ]))
        self.K_mat = K_mat
        return K_mat

    def map_points(self, K_mat):
        for i in range(self.data.shape[-1]):
            x = self.data[:, i].T
            #self.labels_p[i] = np.argmin(np.sqrt(np.sum(np.square(K_mat-x), axis=1)))
            self.labels_p[i] = np.argmin(self.euclidean_distance(K_mat, x, axis=1))
        return self.labels_p

    def euclidean_distance(self, p0, p1, axis=1):
        return np.sqrt(np.sum(np.square(p0-p1), axis=axis))
