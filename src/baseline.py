from torch import nn

'''
AE we learned in class
'''

class Baseline(nn.Module):

    def __init__(self, ch1=6, ch2=12, stride=1, hidden=64, classes=11):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, ch1, 5, stride)
        self.conv2 = nn.Conv2d(ch1, ch2, 5, stride)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(157 * 117 * ch2, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 157 * 117 * 12)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
