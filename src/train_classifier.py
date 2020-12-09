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

def get_accuracy(model, data_loader, auto_encoder__path):
    """
    calcualte accuracy
    """
    auto_model = AutoEncoder().to('cuda')
    auto_model.load_state_dict(torch.load(auto_encoder__path))
    auto_model.eval()
    auto_model = auto_model.encoder
    model = model.to('cuda')
    correct = 0
    total = 0
    if torch.cuda.is_available():
        # print("Using GPU for accuracy calculation")
        use_CUDA = True
    else:
        use_CUDA = False

    for images, labels in iter(data_loader):
        images = images.float()
        images = images.to('cuda')
        num_labels = []
        for i in range(len(labels)):
            idx = int(labels[i][1:]) - 1
            num_labels.append(idx)
        num_labels = torch.tensor(num_labels).long().to('cuda')
        
        embedding = auto_model(images)
        outputs = model(embedding)
        outputs = F.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
        
        #print(model_out)
        #print(pred)
        correct += pred.eq(num_labels.view_as(pred)).sum().item()
        #print(labels)
        #print(correct)
        total += images.shape[0]

    return correct / total 


def train_classifier(
    model,
    auto_encoder__path='./weights/autoencoder.pth',
    learning_rate=0.05,
    data_class=DataClass.TRAINING_SET,
    batch_size=64,
    num_epochs=250,
    plot=False,
    perform_validation=True,
    **kwargs
):
    torch.manual_seed(1000)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('GPU detected!')
        model.to(device)
        # CUDA_LAUNCH_BLOCKING=1


    
    auto_model = AutoEncoder().to('cuda')
    auto_model.load_state_dict(torch.load(auto_encoder__path))
    auto_model.eval()
    
    auto_model = auto_model.encoder


    training_losses = []
    validation_losses = []
    training_acc = []
    validation_acc = []

    criterion = nn.CrossEntropyLoss()
    lsm = nn.LogSoftmax(dim=1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    dataset = HandGesturesDataset(data_class, return_label=True)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    if perform_validation:

        validation_dataset = HandGesturesDataset(DataClass.VALIDATION_SET, return_label=True)
        validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=batch_size)

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
                idx = int(labels[i][1:]) - 1
                num_labels.append(idx)
            num_labels = torch.tensor(num_labels).long().to('cuda')
            num_images = images.shape[0]

            
            embedding = auto_model(images)
            outputs = model(embedding)
            outputs = F.softmax(outputs, dim=1)
            
            # outputs = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, num_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_training_loss += loss.data*num_images
        


        training_losses.append(current_training_loss/num_training_images)
        training_acc.append(get_accuracy(model, dataloader, auto_encoder__path))
        validation_acc.append(get_accuracy(model, validation_dataloader, auto_encoder__path))
        print("train_acc = {}, valid_acc = {}".format(training_acc[-1], validation_acc[-1]))
        print("loss = {}".format(training_losses[-1]))

        current_validation_loss = 0
        num_validation_images = validation_dataset.__len__()

    torch.save(model.state_dict(), './weights/classifier_embed_20.pth')

    plt.plot(training_losses, label = "training_losses")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training_Loss')
    plt.legend()
    plt.show()

    plt.plot(training_acc, label = "training_acc")
    plt.plot(validation_acc, label = "validation_acc")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()


    


if __name__ == '__main__':
    model = Classifier()
    # train_classifier(model, auto_encoder__path='./weights/autoencoder_embed_20.pth')
    train_classifier(model)
