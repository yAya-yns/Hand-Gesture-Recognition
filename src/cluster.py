import umap
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

from mpl_toolkits.mplot3d import Axes3D

with open('./encodings.npy', 'rb') as f:
    embeddings = np.load(f, allow_pickle=True)

encoder = umap.UMAP(n_components=3)
embeddings = encoder.fit_transform(embeddings)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(embeddings[:,0],embeddings[:,1],embeddings[:,2])

clusterer = hdbscan.HDBSCAN()
clusterer.fit(embeddings)

