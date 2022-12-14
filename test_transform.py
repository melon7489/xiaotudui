from torchvision import transforms
from torch import nn

# 1.transformers如何使用
# 2.tensor是一个什么东西 -->张量，里面包含了深度学习相关的属性，方法（反向传播方法，设备选择）
import cv2
img_path_ants = "dataset/train/ants/0013035.jpg"
img = cv2.imread(img_path_ants)
img_trans = transforms.ToTensor()  # 1.创建所需要的工具类对象
img_tensor = img_trans(img)  # 2.调用对象，传入参数，获得结果
print(type(img_tensor))
