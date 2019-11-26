import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=32)
        self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=64)

        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # print(type(x))
        x = self.cnn_1(x)
        # print(x.size())
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.cnn_2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        print(x.size())
        x, _ = torch.max(x, 2)
        print(x.size())
        x = self.fc(x)

        return x


cnn = CNN()

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.,),(1.,))])),
                                           batch_size=16,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',
                                                         train=False,
                                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                                       transforms.Normalize((0.,),
                                                                                                            (1.,))])),
                                          batch_size=16,
                                          shuffle=True)

epoch_num = 10
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(cnn.parameters(), lr=0.001)

for epoch in range(epoch_num):
    cnn.train()
    for index, (train_x, train_y) in enumerate(train_loader):
        cnn.zero_grad()
        optim.zero_grad()
        output = cnn(train_x)
        loss = criterion(output, train_y)

        loss.backward()
        optim.step()
        print(loss.item())

    cnn.eval()
