'''
Documentation, License etc.

@package kmeans_and_em
'''
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import plotly.express as px


from src.KMeans import KMeans
from src.GaussianDistributionGenerator import generate_gaussian_data

"Data Parameters"
K = 4
dims = 2 # Dimensions of Data
samples = 100000 # Number of samples in each distribution

"Distribution Parameters"
# Number of means to search for, also number of distributions to create
priors = npr.dirichlet(np.ones(K), size=1) # Random Priors
priors_csum = np.r_[0, np.cumsum(priors)]
pd = npr.rand(samples) # Decision vector for choosing observed data based on priors

"Data matrix"
data_K = np.zeros((K,dims,samples)) # All distribution data
data = np.zeros((dims,samples)) # Observation data
labels = np.zeros((samples), dtype=int)
means = np.zeros((K, dims))
cov = np.zeros((K, dims, dims))

"Generate K Gaussian Distributions"
for k in range(K):
    means[k, ] = npr.rand(dims)*npr.randint(1, high=10)
    cov[k, ] = npr.rand(dims,dims)*npr.randint(1, high=3)
    data_K[k, ] = npr.multivariate_normal(means[k, ], cov[k, ], size=samples).T

    observation = np.logical_and(pd>priors_csum[k], pd<=priors_csum[k+1])
    data = np.where(observation, data_K[k, ], data)
    labels = np.where(observation, k, labels)

km = KMeans(K, data, labels)

K_mat = km.generate_means(True)
print(K_mat)

labels_mapped = km.map_points(K_mat)
print(labels_mapped)

"Plot 2-D data"
legend_list = []
if data.shape[0] == 2:
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True)
    #plt.figure()
    for k in range(K):
        #ipdb.set_trace()
        axs[0].plot(data_K[k, 0, :], data_K[k, 1, :], '.', alpha=0.50)
        axs[1].plot(data[0, np.where(labels == k)].squeeze(),
                 data[1, np.where(labels == k)].squeeze(),
                 '.',
                 alpha=0.50
                 )

        [axs[a].plot(means[k, 0], means[k, 1], '.', ms=10, zorder=K+1) for a in range(2)]
        #axs[1].plot(means[k, 0], means[k, 1], '.', ms=10, zorder=K+1)
        #legend_list.append("L"+str(k)+" Data")
        #legend_list.append("L"+str(k)+" Mean")

        #plt.legend(legend_list)
    #ipdb.set_trace()
    [axs[a].legend(sum([["L"+str(l)+" Data","L"+str(l)+" Mean"] for l in range(K)], [])) for a in range(2)]
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
