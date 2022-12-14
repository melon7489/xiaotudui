from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter("logs")  # 存放事件文件的路径  在命令行中输入tensorboard --logdir=logs
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)  # 输入数据画图  参数为1.表名 2.y轴数值 3.x轴数值

img_path_ants = "dataset/train/ants/0013035.jpg"
img_path_bees = "dataset/train/bees/16838648_415acd9e3f.jpg"
img_ant = cv2.imread(img_path_ants)
img_bee = cv2.imread(img_path_bees)
writer.add_image("蚂蚁", img_ant, 1, dataformats='HWC')  # 添加图片  参数1.图题 2.图像 3.哪一步  图题相同，步数不同，放在同一个表内
writer.add_image("蜜蜂", img_bee, 1, dataformats='HWC')  # 图题不同，放在不同表内
writer.close()
