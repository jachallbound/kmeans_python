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
K = 2
priors = npr.dirichlet(np.ones(K), size=1) # Random Priors
priors_csum = np.r_[0, np.cumsum(priors)]
pd = npr.rand(samples) # Decision vector for choosing observed data based on priors

"Data matrix"
data_K = np.zeros((K,dims,samples)) # All distribution data
data = np.zeros((dims,samples)) # Observation data
labels = np.zeros((1,samples), dtype=int)
means = np.zeros((K, dims))
cov = np.zeros((K, dims, dims))

"Generate K Gaussian Distributions"
for k in range(K):
    means[k, ] = npr.rand(dims)*npr.randint(1, high=5)
    cov[k, ] = npr.rand(dims,dims)*npr.randint(1, high=2)
    data_K[k, ] = npr.multivariate_normal(means[k, ], cov[k, ], size=samples).T

    observation = np.logical_and(pd>priors_csum[k], pd<=priors_csum[k+1])
    data = np.where(observation, data_K[k, ], data)
    labels = np.where(observation, k, labels)

km = KMeans(K, data, labels)

K_mat = km.get_means(True)
print(K_mat)

labels_mapped = km.map_points(K_mat)
print(labels_mapped)

"Plot 2-D data"
if data.shape[0] == 2:
    plt.figure()
    for k in range(K):
        plt.plot(data_K[k, 0, :], data_K[k, 1, :], '.', alpha=0.50)
    plt.legend([str(l) for l in range(K)])
    plt.draw()
"Plot histograms of labels and labels_mapped"
#fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#axs[0].hist(np.sort(labels.squeeze()), bins=K, density=True)
#axs[1].hist(np.sort(labels_mapped.squeeze()), bins=K, density=True)
#plt.draw()


plt.show(block=False)
#fig = px.scatter_3d(x=data[0], y=data[1], z=data[2])
#fig.show()


#data = npr.rand(100,100,100)*100
