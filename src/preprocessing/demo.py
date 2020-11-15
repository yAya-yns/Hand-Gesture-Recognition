import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import os

from src import model
from src.hand import Hand

'''
Code to get poses

You should not need to run this because I already ran the poses which
are saved in this repo. Look at train.py for what
to do next.

If you need to generate new poses, download the pose repo:

https://github.com/Hzzone/pytorch-openpose

Place this in the root folder (where demo.py is)

Drag/drop your images (in the format generated
by the dataset splitting) into the images folder
where there should be demo images when you
download the repo
'''

image_dirs = os.listdir('./images')

hand_estimation = Hand('model/hand_pose_model.pth')

for dir in image_dirs:
    for image_name in os.listdir(f'./images/{dir}'):
        image = cv2.imread(f'./images/{dir}/{image_name}')

        peaks = hand_estimation(image)

        image_name = ''.join(image_name.split('.'))[:-1] + '.npy'
        with open(f'./poses/{dir}/{image_name}', 'wb') as f:
            np.save(f, peaks)

        break
    break

