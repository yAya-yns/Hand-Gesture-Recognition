import torch
import matplotlib.pyplot as plt

def get_accuracy(model, data_loader):
    """
    calcualte accuracy
    """
    correct = 0
    total = 0
    if torch.cuda.is_available():
        print("Using GPU for accuracy calculation")
        use_CUDA = True
    else:
        use_CUDA = False

    for images, labels in iter(data_loader):
        if use_CUDA:
            images = images.cuda()
            labels = labels.cuda()
        
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
    
    if torch.cuda.is_available():
        use_CUDA = True
        print("Using GPU for training")
    else:
        use_CUDA = False

    for epoch in range(num_epochs):
        print("epoch", epoch)
        for i, (images, labels) in enumerate(train_loader):
            if i % 5 == 0:
                print(i * batch_size)

            if use_CUDA:
                # print("using GPU")
                images = images.cuda()
                labels = labels.cuda()

            model_out = model(images)
            # print(model_out)
            # print(labels)
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