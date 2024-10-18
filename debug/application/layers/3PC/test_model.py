"""
AlexNet for MNIST dataset
"""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x
