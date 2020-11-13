import torch
import pandas as pd
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from dataloader import DataClass, HandGesturesDataset
from model import AutoEncoder

def inference(data_type=DataClass.TRAINING_SET):

    model_path = './weights/autoencoder.pth'
    model = AutoEncoder().to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    encodings = []

    dataset = DataLoader(HandGesturesDataset(
        data_type=data_type, return_label=True))

    for data, label in dataset:

        encodings.append([label[0]] + list(model.encoder(torch.Tensor(data.float()).cuda()).cpu().detach().numpy()[0]))

    with open('./encodings.npy', 'wb') as f:
        np.save(f,np.vstack(encodings))

if __name__ == '__main__':
    inference()




