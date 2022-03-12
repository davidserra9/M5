import torch
from torch import nn

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(3,32,kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(3,3))

        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))

        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.3)

        self.linear1 = nn.Linear(64,64)
        self.linear2 = nn.Linear(64,8)

        self.globalAvgPooling = nn.AvgPool2d((62,62))

        self.softmax = nn.Softmax()


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)


        x = self.globalAvgPooling(x)
        x = torch.squeeze(x)
        x = torch.flatten(x,1)

        """ x = self.linear1(x) """
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.softmax(x)
        return x