import torch
from torch import nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class TuDui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)  # 卷积层
        self.maxpool = nn.MaxPool2d(3, 2, ceil_mode=True)  # ceil_mode = True时对pooling核超出的部分也不会丢弃
        self.relu = nn.ReLU()  # 非线性激活
        self.flatten = nn.Flatten()  # 摊平，拉成一个向量.1.开始维度 2.结束维度
        self.liner = nn.Linear(768, 10)  # 线性层 = 全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.liner(x)
        return x


# 创建模型对象
tudui = TuDui()
print(tudui)
# 创建数据集对象
dataset = torchvision.datasets.CIFAR10("./CIFAR10", False, torchvision.transforms.ToTensor())
# 加载数据集对象
data_loder = DataLoader(dataset, 64, True)
# tensorboard
writer = SummaryWriter("./nnmodulelogs")
for i, data in enumerate(data_loder):
    imgs, labels = data
    print("inputshape:", imgs.shape)
    output = tudui(imgs)
    output = torch.reshape(output, [-1, 1, 10, 1])
    print("outputshape:", output.shape)
    writer.add_images("in", imgs, i)
    writer.add_images("out", output, i)
writer.close()
