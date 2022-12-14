from collections import OrderedDict
import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# 搭建模型基本过程
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.liner1 = nn.Linear(1024, 64)
        self.liner2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.liner1(x)
        x = self.liner2(x)
        return x


''' 
利用sequential改造、简化模型 
'''


class CifarSequential(nn.Module):
    def __init__(self):
        super(CifarSequential, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10("./CIFAR10", False, torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, 4, True)
    loss = nn.CrossEntropyLoss()

    cifar = CifarSequential()
    optim = torch.optim.SGD(cifar.parameters(), 0.001)  # 配置优化器
    for epoch in range(2):
        total_loss = 0
        for data in dataloader:
            imgs, targets = data
            output = cifar(imgs)
            binary_loss = loss(output, targets)
            total_loss = total_loss + binary_loss
            optim.zero_grad()  # 梯度清0，防止上一步干扰
            binary_loss.backward()
            optim.step()  # 调用优化器，对模型参数进行调整
        print(total_loss)