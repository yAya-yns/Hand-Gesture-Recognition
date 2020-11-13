import umap
import numpy as np
import matplotlib.pyplot as plt
import hdbscan

from mpl_toolkits.mplot3d import Axes3D

def get_colors(labels):

    unique_labels = set(labels)
    colors = np.random.rand(len(unique_labels),3)

    color_dict = {}

    for i, label in enumerate(unique_labels):
        color_dict[label] = tuple(colors[i])
    return [color_dict[label] for label in labels]

with open('./encodings.npy', 'rb') as f:
    file = np.load(f, allow_pickle=True)

embeddings = file[:,1:]
labels = file[:,0]

colors = get_colors(labels)

encoder = umap.UMAP(n_components=3)
embeddings = encoder.fit_transform(embeddings)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


'''
clusterer = hdbscan.HDBSCAN()
clusterer.fit(embeddings)

colors = [color_dict.get(color, 'k') for color in clusterer.labels_]
'''

ax.scatter(embeddings[:,0],embeddings[:,1],embeddings[:,2], c=colors)

adjacency_matrix =

