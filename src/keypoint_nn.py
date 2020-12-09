import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataloader import DataClass, HandGesturesDataset
from torch.utils.data import Dataset, DataLoader
from custom_test_keypoints import CustomTestDataset
from torchsummary import summary
import numpy as np

'''
directly train a classifier on keypoints as a baseline
'''

class nn_baseline(nn.Module):

    def __init__(self, hidden_size = 32 ):
        super(nn_baseline, self).__init__()
        self.fc1 = nn.Linear(42, hidden_size) # keypoint input size is 42 ?
        self.fc2 = nn.Linear(hidden_size, 11)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_accuracy(model, data_loader):
    """
    calcualte accuracy
    """
    correct = 0
    total = 0

    for images, labels in iter(data_loader):
        images = images.float()
        model_out = model(images)
        pred = model_out.max(1, keepdims=True)[1]
        # print(model_out)
        print("pred: ", pred.squeeze(), "\nlabel: ", labels)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        # print("Label: ", labels)
        # print(correct)
        total += images.shape[0]

    return correct / total

def get_pred(net, loader):
  pred = []
  label = []
  for i, data in enumerate(loader, 0):
    inputs, labels = data
    inputs = inputs.float()
    outputs = net(inputs)
    outputs = outputs.max(1, keepdims=True)[1]
    outputs = outputs.numpy()
    labels = labels.numpy()
    pred.append(outputs.reshape((1, outputs.size)))
    label.append(labels.reshape((1, labels.size)))

  return pred, label

def evaluate(net, loader, criterion):
    """ Evaluate the network on the loader set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader\
         criterion: The loss function
     Returns:
         acc: A scalar for the avg classification accuracy over the loader set
         loss: A scalar for the average loss function over the loader set
     """
    total_loss = 0.0
    total_acc = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.float()

        # calculate loss
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # calculate accuracy
        pred = outputs.max(1, keepdim=True)[1]
        total_acc += pred.eq(labels.view_as(pred)).sum().item()
        total_epoch += len(labels)
    acc = float(total_acc) / total_epoch
    loss = float(total_loss) / (i + 1)
    return acc, loss

def plot_training_curve(train_err, val_err, train_loss, val_loss):
    """ Plots the training curve for a model run, given  the train/validation
    accuracy/loss.
    """
    import matplotlib.pyplot as plt
    plt.title("Train vs Validation Accuracy")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def train(model, train_loader, val_loader, batch_size=64, num_epochs=1, learn_rate=0.0005):
    """
    train model
    """

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print("epoch", epoch)
        for i, (images, labels) in enumerate(train_loader):

            images = images.float()
            model_out = model(images)
            loss = criterion(model_out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_acc[epoch], train_loss[epoch] = evaluate(model, train_loader, criterion)
        val_acc[epoch], val_loss[epoch] = evaluate(model, val_loader, criterion)

        print('train acc: ' + str(train_acc[-1]) + ' | train loss: ' + str(float(loss)) + ' | valid acc: ' + str(
            val_acc[-1]))

    return train_acc, val_acc, train_loss, val_loss



if __name__ == '__main__':
    batch_size = 64
    train_set = HandGesturesDataset(DataClass.TRAINING_SET, return_label=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    val_set = HandGesturesDataset(DataClass.VALIDATION_SET, return_label=True)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size)

    test_set = HandGesturesDataset(DataClass.TEST_SET, return_label=True)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    nn_baseline = nn_baseline()
    #summary(nn_baseline, input_size=(1, 42))
    #train_acc, val_acc, train_loss, val_loss = train(nn_baseline, train_loader, val_loader, batch_size=64, num_epochs=120, learn_rate=0.005)
    #torch.save(nn_baseline.state_dict(), 'keypoint_nn_snapshot')
    #plot_training_curve(train_acc, val_acc, train_loss, val_loss)
    state = torch.load('keypoint_nn_snapshot')
    nn_baseline.load_state_dict(state)
    print(get_accuracy(nn_baseline, test_loader))
    pred, label = get_pred(nn_baseline, test_loader)
    print(pred, label)

    custom_test_dataset = CustomTestDataset()
    cus_test_loader = DataLoader(custom_test_dataset, batch_size=64, shuffle=True)
    print(get_accuracy(nn_baseline, cus_test_loader))
    pred, label = get_pred(nn_baseline, cus_test_loader)
    print(pred, label)


