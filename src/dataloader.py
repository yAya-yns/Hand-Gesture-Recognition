import os
import shutil
import cv2
import torch
import numpy as np
import pickle

from torch import nn
from enum import Enum
# from PIL import Image
from torch.utils.data import Dataset, DataLoader

class DataClass(Enum):
    TRAINING_SET = './data/poses/train'
    VALIDATION_SET = './data/poses/validation'
    TEST_SET = './data/poses/test'

class HandGesturesDataset(Dataset):
    """Hand Gestures Dataset."""

    def __init__(
        self,
        data_type
    ):

        self.data_type = data_type
        self.file_list = os.listdir(self.data_type.value)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        name = self.file_list[idx]
        with open(f'{self.data_type.value}/{name}', 'rb') as f:
            keypoints = np.load(f)

        keypoints = (keypoints - keypoints.min(axis=0))\
            /(keypoints.max(axis=0)-keypoints.min(axis=0))
        return keypoints.flatten()


if __name__ == '__main__':

    dataset = HandGesturesDataset(DataClass.TRAINING_SET)
    data_loader = DataLoader(dataset)



