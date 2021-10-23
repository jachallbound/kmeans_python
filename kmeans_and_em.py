'''
Documentation, License etc.

@package kmeans_and_em
'''
import numpy as np
import numpy.random as npr
import plotly.express as px
import matplotlib.pyplot as plt

from src.KMeans import KMeans

"Data Parameters"
dims = 2 # Dimensions of Data
samples = 100000 # Number of samples in each distribution

"Distribution Parameters"
# Number of means to search for, also number of distributions to create
K = 4
priors = npr.dirichlet(np.ones(K), size=1) # Random Priors
priors_csum = np.r_[0, np.cumsum(priors)]
pd = npr.rand(samples) # Decision vector for choosing observed data based on priors

"Data matrix"
data_K = np.zeros((K,dims,samples)) # All distribution data
data = np.zeros((dims,samples)) # Observation data
labels = np.zeros((1,samples))
means = np.zeros((K, dims))
cov = np.zeros((K, dims, dims))

"Generate K Gaussian Distributions"
for k in range(K):
    means[k, ] = npr.rand(dims)*npr.randint(1, high=10)
    cov[k, ] = np.eye(dims)*npr.randint(1, high=10)
    data_K[k, ] = npr.multivariate_normal(means[k, ], cov[k, ], size=samples).T

    observation = np.logical_and(pd>priors_csum[k], pd<=priors_csum[k+1])
    data = np.where(observation, data_K[k, ], data)
    labels = np.where(observation, k, labels)

    #data[priors_csum[k] < pd <= priors_csum[k+1]] = 1


km = KMeans(K, data)

K_mat = km.get_means(True)

print(K_mat)

#fig = px.scatter_3d(x=data[0], y=data[1], z=data[2])
#fig.show()


#data = npr.rand(100,100,100)*100
