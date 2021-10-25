import copy
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt

def SubPlotData(K, data, labels2, means2):
    if data.shape[0] == 2:
        proj = "rectilinear"
    elif data.shape[0] == 3:
        proj = "3d"
    else:
        return -1

    "Generate plots with more than 10 colors"
    "Deepcopy or else you change the colors matplotlib calls"
    colors = copy.deepcopy(matplotlib.colors.get_named_colors_mapping())
    [colors.popitem() for i in range(10)] # Remove starting colors that aren't as good

    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True,
                            subplot_kw={'projection': proj})
    for k in range(K):
        if data.shape[0] == 2:
            c = colors.popitem()[1]
            [axs[a].plot(data[0, np.where(labels2[a] == k)].squeeze(),
                        data[1, np.where(labels2[a] == k)].squeeze(),
                        '.',
                        alpha=0.50,
                        color=c)
                        for a in range(2)]
            c = colors.popitem()[1]
            [axs[a].plot(means2[a, k, 0],
                         means2[a, k, 1],
                         '.',
                         ms=10,
                         zorder=K+1,
                         color=c) for a in range(2)]
        elif data.shape[0] == 3:
            c = colors.popitem()[1]
            [axs[a].plot(data[0, np.where(labels2[a] == k)].squeeze(),
                        data[1, np.where(labels2[a] == k)].squeeze(),
                        data[2, np.where(labels2[a] == k)].squeeze(),
                        '.',
                        alpha=0.50,
                        color=c)
                        for a in range(2)]
            c = colors.popitem()[1]
            [axs[a].plot(means2[a, k, 0],
                         means2[a, k, 1],
                         means2[a, k, 2],
                         '.',
                         ms=10,
                         zorder=K+1,
                         color=c) for a in range(2)]
    [axs[a].legend(sum([["L"+str(l)+" Data","L"+str(l)+" Mean"] for l in range(K)], [])) for a in range(2)]
    plt.draw()
