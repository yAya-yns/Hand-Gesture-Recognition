import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from model import AutoEncoder
from dataloader import DataClass, HandGesturesDataset
from torch.utils.data import Dataset, DataLoader

torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''
How to train:

Make sure that there is a data folder in the same folder. In this folder
should be the following path:

data/poses/train/

in train have all the npy files for gestures

This file is in the repo - however you need to unzip the files in the
data folder
'''

def train_model(
    model,
    learning_rate=0.5,
    data_class=DataClass.TRAINING_SET,
    batch_size=64,
    num_epochs=250,
    plot=False,
    perform_validation=True,
    **kwargs
):

    '''
    Here is the model training code.

    Nothing too interesting here.
    '''


    #Train on CPU if one available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available:
        print('GPU detected!')

    model.to(device)

    training_losses = []
    validation_losses = []

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    dataset = HandGesturesDataset(data_class)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    if perform_validation:

        validation_dataset = HandGesturesDataset(DataClass.VALIDATION_SET, **kwargs)
        validation_dataloader = DataLoader(validation_dataset, batch_size=64)

    config = [
        model.__class__.__name__,
        str(learning_rate),
        data_class._name_,
        str(batch_size)
    ]

    for i in range(num_epochs):

        print(f'Epoch {i}')

        current_training_loss = 0
        num_training_images = dataset.__len__()

        for images in dataloader:

            images = images.float()

            num_images = images.shape[0]

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            current_training_loss += loss.data*num_images
        print(current_training_loss)

        training_losses.append(current_training_loss/num_training_images)

        current_validation_loss = 0
        num_validation_images = validation_dataset.__len__()

    plt.plot(training_losses)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), './weights/autoencoder.pth')

if __name__ == '__main__':


    model = AutoEncoder()

    train_model(model)

