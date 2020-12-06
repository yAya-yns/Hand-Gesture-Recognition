import umap
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import pickle

from scipy import spatial
from mpl_toolkits.mplot3d import Axes3D
from dataloader import DataClass, HandGesturesDataset
from torch.utils.data import Dataset, DataLoader
'''
Clustering algo.

Idea:

Take latent space. Use UMAP to compress down to 3 dimensions
For each cluster, since we know the labels we can find cluster centroids
and uncertainty ellipsoids.

For each uncertainty ellipsoid, denoise by removing a small portion of it

Run this after running inference.py

'''

def denoise_pointcloud(points):

    '''
    Similar to RANSAC (but lazier)
    Remove outliers by looking at neighborhood densities for points
    '''


    rejection_thresh = .8
    num_points_to_keep = int(points.shape[0]*rejection_thresh)
    delta = 2

    tree = spatial.cKDTree(points)
    local_densities = []

    for point in points:

        indices = tree.query_ball_point(point, delta)
        local_density = np.linalg.norm(points[indices] - point)

        local_densities.append(local_density)

    indices_to_keep = np.argsort(local_densities)[:num_points_to_keep]


    return points[indices_to_keep]


def plot_detection_ellipsoid(cluster_dict, ax, test_only=False):

    for k, v in cluster_dict.items():

        if test_only:
            if 'T' not in k:
                continue

        coefs = v['std']
        centroid = v['centroid']

        rx, ry, rz = coefs

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v)) + centroid[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) + centroid[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + centroid[2]

        # Plot:
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)

        # Adjustment of the axes, so that they all have the same span:
        max_radius = max(rx, ry, rz)
        #for axis in 'xyz':
        #    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))


def assign_clusters(labels, embeddings):

    unique_labels = set(labels)
    cluster_dict = {}

    for label in unique_labels:

        points = embeddings[labels == label]

        points = denoise_pointcloud(embeddings[labels == label])

        centroid = points.mean(axis=0)
        std = points.std(axis=0)

        cluster_dict[label] = {}
        cluster_dict[label]['centroid'] = centroid
        cluster_dict[label]['std'] = std

    return cluster_dict

def get_colors(labels):

    unique_labels = set(labels)
    colors = np.random.rand(len(unique_labels),3)

    color_dict = {}

    for i, label in enumerate(unique_labels):
        color_dict[label] = tuple(colors[i])
    return [color_dict[label] for label in labels]

if __name__ == '__main__':

    plot_test_only = True

    data_class = DataClass.TRAINING_SET
    batch_size = 64

    dataset = HandGesturesDataset(data_class, return_label=True)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    keypoints = []
    all_labels = []

    dataset_custom = HandGesturesDataset(DataClass.CUSTOM, return_label=True)
    dataloader_custom = DataLoader(dataset_custom, shuffle=True, batch_size=batch_size)

    for images, labels in dataloader:

        #print(labels, images)
        keypoints.append(images.numpy())
        all_labels.append(labels)

    for images, labels in dataloader_custom:

        #print(labels, images)
        keypoints.append(images.numpy())
        all_labels.append(labels)


    #with open('./data/encodings.npy', 'rb') as f:
        #file = np.load(f, allow_pickle=True)

    embeddings = np.concatenate(keypoints)
    labels = np.concatenate(all_labels)

    colors = get_colors(labels)

    encoder = umap.UMAP(n_components=3)
    embeddings = encoder.fit_transform(embeddings)

    cluster_dict = assign_clusters(labels, embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if plot_test_only:
        mask = [True if 'T' in label else False for label in labels]

        embeddings = embeddings[mask]
        colors = np.array(colors)[mask]

    ax.scatter(embeddings[:,0],embeddings[:,1],embeddings[:,2], c=colors)
    ax.set(xlim=(-10,10), ylim=(-10,10), zlim=(-10,10))

    for k, v in cluster_dict.items():

        if plot_test_only:
            if 'T' not in k:
                continue

        x, y, z = v['centroid']
        ax.scatter(x, y, z, s=50, c='r')


    plot_detection_ellipsoid(cluster_dict, ax, test_only=plot_test_only)

    plt.show()

    with open('./encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    with open('./label_dict.pkl', 'wb') as f:
        pickle.dump(cluster_dict, f)
