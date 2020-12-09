import os
import shutil
import cv2
import torch
import numpy as np
import pickle

from torch import nn
from enum import Enum
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader

CUSTOM_TEST_SET = './data/custom_test/'

class CustomTestDataset(Dataset):
    "Custom Test Dataset"

    def __init__(self):
        self.dir = CUSTOM_TEST_SET
        self.file_list = os.listdir(self.dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        with open(f'{self.dir}/{name}', 'rb') as f:
            keypoints = np.load(f)

        keypoints = (keypoints - keypoints.min(axis=0))\
            /(keypoints.max(axis=0)-keypoints.min(axis=0))

        return keypoints.flatten(), int(name.split('_')[0][1:]) - 1

if __name__ == '__main__':
    dataset = CustomTestDataset()
    test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    dataiter = iter(test_loader)
    keypoints, labels = dataiter.next()
    print(keypoints, labels)
