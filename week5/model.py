import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.fc3 = nn.Linear(84, 1)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""
net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
"""

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

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))

        self.bnorm1 = nn.BatchNorm2d(num_features=32)
        self.bnorm2 = nn.BatchNorm2d(num_features=64)

        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.3)

        self.globAvgPool = nn.AdaptiveAvgPool2d((1,1))

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


if __name__ == "__main__":
    """
    Tests to see if there are errors
    """
    x = torch.randn((16, 3, 224, 224))      # 16 random images
    model = NetSquared()                    # model init
    preds = model(x)                        # model predictions
