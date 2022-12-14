import torch
import torchvision


vgg16 = torchvision.models.vgg16(False)
# 保存/加载模型，方式1
'''注意：用此方法加载模型时，模型必须先定义好（不用创建对象）'''
torch.save(vgg16, "./vgg16.pt")  #保存
model = torch.load("./vgg16.pt")  # 加载
print(model)

# 保存/加载模型，方式2
torch.save(vgg16.state_dict(), "./vgg16_weight.pt")  # 保存（只有权重）
vgg16 = torchvision.models.vgg16(False)  # 加载
vgg16.load_state_dict(torch.load("./vgg16_weight.pt"))
print(vgg16)
