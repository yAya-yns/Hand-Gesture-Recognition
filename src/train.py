import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from model import AutoEncoder
from dataloader import DataClass, HandGesturesDataset
from torch.utils.data import Dataset, DataLoader

torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

        #fingerprint = f'./weights/{"-".join(config + [str(i)])}.pth'
        #torch.save(model.state_dict(), fingerprint)

        current_validation_loss = 0
        num_validation_images = validation_dataset.__len__()

        for images in validation_dataloader:

            num_images = images.shape[0]

            images = images.float()

            outputs = model(images)
            loss = criterion(outputs, images)

            #Use loss.data to free up gpu memory
            current_validation_loss += loss.data*num_images

        validation_losses.append(current_validation_loss/num_validation_images)

    plt.plot(validation_losses)
    plt.plot(training_losses)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), './weights/autoencoder.pth')

if __name__ == '__main__':


    model = AutoEncoder()

    train_model(model)


    dataset = HandGesturesDataset(DataClass.TRAINING_SET)

    dataloader = DataLoader(dataset)

    '''

    encodings = []

    for data in dataloader:
        encodings.append(model.encoder(torch.Tensor(data.float()).float()).cpu().detach())

    with open('./encodings.npy', 'wb') as f:
        np.save(f,np.vstack(encodings))

    '''


