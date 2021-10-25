import numpy as np
import numpy.random as npr

def generate_gaussian_data(D,
                           ndim,
                           samples,
                           priors = None,
                           means = None,
                           covs = None
                           ):
    """Generate D N-dimension Gaussian Distributions of size samples.

    Parameters:
    ---
    D: *int*
        Amount of IID Gaussian clusters to create.
        When used in K-Means, should probably be equal to K.
    ndim: *int*
        Dimensionality of data.
        Returned data with be ndim arrays of sample data (in a matrix)
    samples: *int*
        Amount of data samples per Gaussian cluster.
    priors: *array_like*
        D element array of priors of corresponding Gaussian distributions.
        Default: random values from a Dirichlet distribution.
    means: *array_like*
        D by ndim matrix of means.
        Default: rand(ndim) values times randint(1,high=10) (D times)
    covs = *array_like*
        D by ndim by ndim matrix of covariances.
        Default: rand(ndim,ndim) values times randint(1,high=3) (D times)
                This can lead to 'not positive-semidefinite' covs for high ndim.

    Returns:
    ---
    gaussian_dict: *dict*
        Dictionary of data, labels, and Gaussian distribution parameters.
    """

    "Generate distribution parameters if they weren't passed"
    if priors == None:
        priors = npr.dirichlet(np.ones(D), size=1)
    if means == None:
        means = np.array([npr.rand(ndim)*npr.randint(1,high=10) for i in range(D)])
    if covs == None:
        covs = np.array([npr.rand(ndim,ndim)*npr.randint(1, high=3) for i in range(D)])

    "Distribution Parameters"
    # Number of means to search for, also number of distributions to create
    priors_csum = np.r_[0, np.cumsum(priors)]
    pd = npr.rand(samples) # Decision vector for choosing observed data based on priors

    "Data matrix"
    data_D = np.zeros((D,ndim,samples)) # All distribution data
    data = np.zeros((ndim,samples)) # Observation data
    labels = np.zeros((samples), dtype=int)

    "Generate K Gaussian Distributions"
    for d in range(D):
        data_D[d, ] = npr.multivariate_normal(means[d, ], covs[d, ], size=samples).T

        observation = np.logical_and(pd>priors_csum[d], pd<=priors_csum[d+1])
        data = np.where(observation, data_D[d, ], data)
        labels = np.where(observation, d, labels)

    gaussian_dict = {'data': data,
                        'labels': labels,
                        'data_D': data_D,
                        'priors': priors,
                        'means': means,
                        'covs': covs}
    return gaussian_dict
