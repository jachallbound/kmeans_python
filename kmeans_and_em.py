'''
Documentation, License etc.

@package kmeans_and_em
'''
import numpy as np
import plotly.express as px

from src.KMeans import KMeans

K = 4
data = np.random.rand(100,100,100)*100
k = KMeans(K, data)

K_mat = k.get_means(True)

print(K_mat)

#fig = px.scatter_3d(x=data[0], y=data[1], z=data[2])
#fig.show()
