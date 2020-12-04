import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from model import AutoEncoder
from model import Classifier
from dataloader import DataClass, HandGesturesDataset
from torch.utils.data import Dataset, DataLoader

def train_classifier(
    model,
    auto_encoder__path='./weights/autoencoder.pth',
    learning_rate=0.5,
    data_class=DataClass.TRAINING_SET,
    batch_size=64,
    num_epochs=250,
    plot=False,
    perform_validation=True,
    **kwargs
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('GPU detected!')
        model.to(device)


    
    auto_model = AutoEncoder().to('cuda')
    auto_model.load_state_dict(torch.load(auto_encoder__path))
    auto_model.eval()
    
    auto_model = auto_model.encoder


    training_losses = []
    validation_losses = []

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    dataset = HandGesturesDataset(data_class, return_label=True)
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

        for images, labels in iter(dataloader):
            images = images.to('cuda')
            images = images.float()
            num_labels = []
            for i in range(len(labels)):
                num_labels.append(int(labels[i][1:]))
            num_labels = torch.tensor(num_labels).to('cuda')
            num_images = images.shape[0]

            optimizer.zero_grad()

            embedding = auto_model(images)
            outputs = model(embedding)
            outputs = F.softmax(outputs)
            outputs = torch.argmax(outputs, dim=1)

            outputs = outputs.to(torch.float32)
            num_labels = num_labels.to(torch.float32)
            
            loss = criterion(outputs, num_labels)
            loss.requires_grad = True 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_training_loss += loss.data*num_images
        print(current_training_loss)

        training_losses.append(current_training_loss/num_training_images)

        current_validation_loss = 0
        num_validation_images = validation_dataset.__len__()

    plt.plot(training_losses)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.show()

    torch.save(model.state_dict(), './weights/classifier.pth')


if __name__ == '__main__':
    model = Classifier()

    train_classifier(model)
