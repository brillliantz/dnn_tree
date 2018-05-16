# encoding: utf-8

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, fc_in_size, output_size, in_channels, dim=2):
        super(Net, self).__init__()

        if dim == 1:
            conv_func = nn.Conv1d
            pool_func = nn.MaxPool1d
        else:
            conv_func = nn.Conv2d
            pool_func = nn.MaxPool2d

        self.output_size = output_size
        self.fc_in_size = fc_in_size
        self.in_channels = in_channels

        self.conv1 = conv_func(in_channels=self.in_channels, out_channels=8,kernel_size=5)
        self.pool1 = pool_func(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = conv_func(in_channels=8, out_channels=16, kernel_size=3)
        self.pool2 = pool_func(kernel_size=2, stride=2)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.conv3 = conv_func(in_channels=16, out_channels=64, kernel_size=3)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=self.fc_in_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        # shapes = x.shape[1:]
        # shape = np.prod(shapes) # 16 * 6 * 6
        shape = self.fc_in_size
        x = x.view(-1, shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def show(self, desc):
        print("\n\n ", desc)
        ps = list(self.parameters())
        print(ps[0][0])


