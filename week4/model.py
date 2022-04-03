import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class NetSquared(nn.Module):
    """
    Class of the custom CNN
    """

    def __init__(self):
        """
        Initialization of the needed layers
        """
        super(NetSquared, self).__init__()

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))

        self.bnorm1 = nn.BatchNorm2d(num_features=32)
        self.bnorm2 = nn.BatchNorm2d(num_features=64)

        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.3)

        self.globAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(in_features=64, out_features=8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxp(x)
        x = self.bnorm1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxp(x)
        x = self.bnorm2(x)
        x = self.dropout2(x)

        x = self.globAvgPool(x)
        x = self.dropout1(x)
        x = torch.squeeze(x)
        x = self.linear(x)

        return x


class EmbeddingNet(nn.Module):
    def __init__(self, backbone):
        super(EmbeddingNet, self).__init__()

        basemodel = torch.hub.load('pytorch/vision:v0.10.0', backbone, pretrained=True)
        self.base_resnet = torch.nn.Sequential(*(list(basemodel.children())[:-1]))

    def forward(self, x):
        output = self.base_resnet(x).squeeze()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


if __name__ == "__main__":
    """
    Tests to see if there are errors
    """
    x = torch.randn((16, 3, 224, 224))  # 16 random images
    model = NetSquared()  # model init
    preds = model(x)  # model predictions
