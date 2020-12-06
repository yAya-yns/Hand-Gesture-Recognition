import umap
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

from scipy import spatial
from mpl_toolkits.mplot3d import Axes3D
from cluster import denoise_pointcloud, get_colors, plot_detection_ellipsoid, assign_clusters

if __name__ == '__main__':

    dir = './data/poses/video/'

    video = os.listdir(dir)

    encodings = []

    with open('label_dict.pkl', 'rb') as f:
        label_dict = pkl.load(f)

    for frame in range(len(video)):

        name = f'frame{frame}jp.npy'

        with open(dir + name, 'rb') as f:
            keypoints = np.load(f).flatten()
            keypoints = (keypoints - keypoints.min(axis=0))\
            /(keypoints.max(axis=0)-keypoints.min(axis=0))
            encodings.append(keypoints)

    with open('./encoder.pkl', 'rb') as f:
        encoder = np.load(f, allow_pickle=True)

    encodings = np.nan_to_num(encodings, 0)
    final_bits = encoder.transform(encodings)

    detected_gestures = []

    for thing in final_bits:

        detected_gesture = None
        candidates = []
        for k, v in label_dict.items():
            if 'T' not in k:
                continue
            if np.sum(np.abs(thing - v['centroid']) < v['std']) == 3:
                detected_gesture = k
        detected_gestures.append(detected_gesture)

    with open('./detected_gestures.npy', 'wb') as f:
        np.save(f, np.array(detected_gestures))



