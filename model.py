import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F


# ***********************************************
# Encoder and Discriminator has same architecture
# ***********************************************
class BrainModel(nn.Module):
    def __init__(self, channel=512, out_class=1):
        super(BrainModel, self).__init__()
        self.channel = channel

        self.conv1 = nn.Conv3d(1, channel // 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel // 8, channel // 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel // 4)
        self.conv3 = nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel // 2)
        self.conv4 = nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)

        self.fc1 = nn.Linear(40960, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1280)


    def forward(self, x):
        output = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        output = F.leaky_relu(self.bn2(self.conv2(output)), negative_slope=0.2)
        output = F.leaky_relu(self.bn3(self.conv3(output)), negative_slope=0.2)
        output = F.leaky_relu(self.bn4(self.conv4(output)), negative_slope=0.2)
        output = output.view((-1, 40960))
        output = F.leaky_relu(self.fc1(output), negative_slope=.2)
        output = F.leaky_relu(self.fc2(output), negative_slope=.2)
        output = F.leaky_relu(self.fc3(output), negative_slope=.2)

        return output