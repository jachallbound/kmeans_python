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
from src.PlotData import SubPlotData

"Data Parameters"
K = 5
dims = 3 # Dimensions of Data
samples = 100000 # Number of samples in each distribution

"Generate Random Data"
gmm_dict = generate_gaussian_data(K, dims, samples)

data =   gmm_dict['data']     # Only need data and labels, but pull out all for error testing
labels = gmm_dict['labels']
data_D = gmm_dict['data_D']
priors = gmm_dict['priors']
means =  gmm_dict['means']
covs =   gmm_dict['covs']

"Create KMeans object with our observed data"
km = KMeans(K, data, labels)

for i in range(1):
    random_means = False
    if i == 0:
        random_means = True
    K_mat = km.generate_means(random_means)
    print(K_mat)

    labels_mapped = km.map_points(K_mat)
    print(labels_mapped)

"Plot 2-D data"
if data.shape[0] == 2|3:
    labels_both = np.stack((labels, labels_mapped))
    means_both = np.stack((means, K_mat))
    SubPlotData(K, data, labels_both, means_both)


"Plot histograms of labels and labels_mapped"
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(np.sort(labels.squeeze()), bins=K, density=True)
axs[1].hist(np.sort(labels_mapped.squeeze()), bins=K, density=True)
plt.draw()


plt.show(block=False)

