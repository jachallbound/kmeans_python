'''
Documentation, License etc.

@package kmeans_and_em
'''
import copy
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

from src.KMeans import KMeans
from src.GaussianDistributionGenerator import generate_gaussian_data
from src.PlotData import SubPlotData

"Data Parameters"
K = 3 # Number of distributions to create and means to classify
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

labels_mapped_prev = np.zeros(labels.shape)
labels_changed_percent = 1.0
i = 0
while labels_changed_percent > 0.01:
#for i in range(4):
    "If first loop, choose random starting means"
    random_means = False
    if i == 0:
        random_means = True

    "Generate matrix of means and map points to them"
    K_mat = km.generate_means(random_means)
    print(f"K Means {K_mat}")
    labels_mapped = km.map_points(K_mat)

    "Check how many points have changed labels since last loop"
    labels_changed_percent = sum(labels_mapped != labels_mapped_prev)/samples
    labels_mapped_prev = copy.deepcopy(labels_mapped)
    i += 1
    print(f"Iteration: {i}, Labels changed: {labels_changed_percent*100:2.2f}%")

"Print detected means and true means"
"Sort means matrices to check mean error"
"TODO: Need to relabel in order to check classification error"
print(f"\n\n")
print(f"K Means\n{np.sort(K_mat, axis=0)}")
print(f"True Means\n{np.sort(means, axis=0)}")
print(f"Mean error {np.mean(km.euclidean_distance(K_mat, means, axis=1)):.2f} (not a percentage)")

"Plot 2-D or 3-D data"
if data.shape[0] == 2 or data.shape[0] == 3:
    labels_both = np.stack((labels, labels_mapped))
    means_both = np.stack((means, K_mat))
    SubPlotData(K, data, labels_both, means_both)


"Plot histograms of labels and labels_mapped"
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(np.sort(labels.squeeze()), bins=K, density=True)
axs[1].hist(np.sort(labels_mapped.squeeze()), bins=K, density=True)
plt.draw()


plt.show(block=False)

