import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集CIFAR
test_dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)  # drop_last为True，对最后一批若不能取到bachsize大小的数据，则删除
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)  # drop_last为True，对最后一批若不能取到bachsize大小的数据，则删除
writer = SummaryWriter("tensorboard")
for epoch in range(2):  # shuffle会对每一个epoch的数据进行打乱
    for i, data in enumerate(train_loader):  # 按照批次取数据
        imgs, labels = data
        writer.add_images("train_loader Epoch:{}".format(epoch), imgs, i)
        print(i)
for epoch in range(2):  # shuffle会对每一个epoch的数据进行打乱
    for i, data in enumerate(test_loader):  # 按照批次取数据
        imgs, labels = data
        writer.add_images("test_loader Epoch:{}".format(epoch), imgs, i)
        print(i)
writer.close()