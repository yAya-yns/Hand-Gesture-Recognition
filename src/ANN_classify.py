import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt

import os
import random
import shutil
import pdb

from dataloader import HandGesturesDataset, DataClass


torch.manual_seed(1)


train_set = HandGesturesDataset(DataClass.TRAINING_SET)
train_loader = DataLoader(train_set)
valid_set = HandGesturesDataset(DataClass.VALIDATION_SET)
valid_loader = DataLoader(valid_set)

class FC_classifier(nn.Module):

    def __init__(self):
        super(FC_classifier, self).__init__()
        self.fc1 = nn.Linear(42, 32)
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(32, 12)

    def forward(self, x):
        x = x.view(-1, 42).float()
        x = nn.functional.relu(self.fc1(x))
        # x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_accuracy(model, data_loader):
    """
    calcualte accuracy
    """
    correct = 0
    total = 0

    for images, labels in iter(data_loader):
        labels = [int(l[1:]) for l in labels]
        labels = torch.from_numpy(np.asarray(labels)).type(torch.LongTensor)
        model_out = model(images)
        pred = model_out.max(1, keepdims=True)[1]
        #print(model_out)
        #print(pred)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        #print(labels)
        #print(correct)
        total += images.shape[0]

    return correct / total 

def train(model, train_loader, val_loader, batch_size=27, num_epochs=1, learn_rate = 0.001):
    """
    train model
    """
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    losses, train_acc, val_acc, iters = [], [], [], []
    

    for epoch in range(num_epochs):
        print("epoch", epoch)
        for i, (images, labels) in enumerate(train_loader):
            # pdb.set_trace()
            labels = [int(l[1:]) for l in labels]
            labels = torch.from_numpy(np.asarray(labels)).type(torch.LongTensor)
            model_out = model(images)
            # pdb.set_trace()
            loss = criterion(model_out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        iters.append(epoch)
        losses.append(float(loss)/batch_size)

        train_acc.append(get_accuracy(model, train_loader))
        val_acc.append(get_accuracy(model, valid_loader))

        print('train acc: ' + str(train_acc[-1]) + ' | train loss: ' + str(float(loss)) + ' | valid acc: ' + str(val_acc[-1]))
        # print(n)
        # if n % 10  == 0:
        #    print(n)
        # n += 1

    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print('done')


myModel = FC_classifier()
myModel = myModel.float()
summary(myModel, (1, 42, 1))
pdb.set_trace()
# train(myModel, train_loader, valid_loader, batch_size=27,  num_epochs=50)

test_set = HandGesturesDataset(DataClass.CUS_TEST_SET)
test_loader = DataLoader(test_set)
train(myModel, train_loader, test_loader, batch_size=27,  num_epochs=50)

pdb.set_trace()