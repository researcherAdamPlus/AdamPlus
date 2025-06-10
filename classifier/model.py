import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(28*28, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward (self, inputs):
        x = self.flatten(inputs)
        x = F.relu(self.dense1(x))
        return self.dense2(x)
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x




# class MNISTClassifier(nn.Module):
#     def __init__(self):
#         super(MNISTClassifier, self).__init__()
#         self.flatten = nn.Flatten()
#         self.dense1 = nn.Linear(28*28, 128)
#         self.dense2 = nn.Linear(128, 128)
#         self.dense4 = nn.Linear(128, 64)
#         self.dense3 = nn.Linear(64, 32)
#         self.dense5 = nn.Linear(32, 10)

#     def forward (self, inputs):
#         x = self.flatten(inputs)
#         x = F.leaky_relu(self.dense1(x))
#         x = F.leaky_relu(self.dense2(x))
#         x = F.leaky_relu(self.dense3(x))
#         x = F.leaky_relu(self.dense4(x))
#         return self.dense5(x)
