import torch
import torchvision.transforms
from PIL import Image
from test_cifarNet import CifarNet

if __name__ == '__main__':
    img_path = "./imgs/airplane.png"
    img = Image.open(img_path)
    trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
    img = trans(img)
    img = torch.reshape(img, [1, 3, 32, 32])
    model = torch.load("./cifar_4_testloss_231.98304748535156.pt")
    model.eval()
    with torch.no_grad():
        output = model(img)
        print(output)
        pred = output.argmax(1)
        print(pred)