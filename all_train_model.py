import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from test_cifarNet import CifarNet
import torch
from torch import nn

'''
如果要转移到GPU上训练需要修改的地方：
1. 数据（输入，标注）
2. 网络模型
3. 损失函数
'''
if __name__ == '__main__':
    # 设置训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 准备数据集
    train_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
    # 看一下数据集的长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集的长度为：{}".format(train_data_size))
    print("测试集的长度为：{}".format(test_data_size))
    # 利用Dataloader加载数据集
    train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
    # 创建网络模型
    cifar = CifarNet()
    '''移植到设备'''
    cifar = cifar.to(device)
    print(cifar)
    # 创建损失函数
    loss_fn = nn.CrossEntropyLoss()
    '''移植到设备'''
    loss_fn = loss_fn.to(device)
    # 创建优化器
    optimizer = torch.optim.SGD(cifar.parameters(), lr=1e-2)
    # 记录训练次数
    total_train_step = 0
    # 记录测试次数
    total_test_step = 0
    # 训练轮数
    epochs = 5
    # 创建tensorboard对象
    writer = SummaryWriter("./logs")
    # 开始训练
    for i in range(epochs):
        # 记录每一轮的总损失
        # 设置模型为训练状态
        cifar.train()
        total_train_loss = 0
        print("-------第 {} 轮训练开始--------".format(i+1))
        for data in train_dataloader:
            imgs, targets = data
            '''移植到设备'''
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = cifar(imgs)
            loss = loss_fn(outputs, targets)
            # 优化器优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 进行训练计数
            total_train_step = total_train_step + 1
            total_train_loss = total_train_loss + loss
            if total_train_step % 100 == 0:
                print("训练次数：{}，损失：{}".format(total_train_step, loss.item()))
                writer.add_scalar("训练集损失", loss.item(), total_train_step)
        print("第 {} 轮训练集的总损失：{}".format(i+1, total_train_loss.item()))
        # 测试步骤开始
        # 设置模型为测试模式
        cifar.eval()
        # 测试集总体损失
        total_test_loss = 0
        # 测试机总体正确率
        total_test_accurcy = 0
        # 设置无梯度，不对网络进行优化
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                outputs = cifar(imgs)
                accurcy = (outputs.argmax(1) == targets).sum()
                total_test_accurcy = total_test_accurcy + accurcy
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss
            print("测试集的正确率为：{}".format(total_test_accurcy/test_data_size))
            writer.add_scalar("测试集正确率", total_test_accurcy/test_data_size, total_test_step)
            print("测试集的损失为：{}".format(total_test_loss))
            writer.add_scalar("测试集损失", total_test_loss, total_test_step)
            total_test_step = total_test_step + 1
        # 保存模型
        torch.save(cifar, "./checkpoint/cifar_{}_testloss_{}.pt".format(i,total_test_loss))
        print("保存模型中........")
    writer.close()