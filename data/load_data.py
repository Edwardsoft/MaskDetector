# 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
import torch
import torchvision
from torchvision import transforms


def load_data_fashion_mnist(batch_size, resize=None,
root='C:\Users\639\maskDectorData\\'):


    trainset = torchvision.datasets.ImageFolder(root, transform=transforms.Compose([
                                                    transforms.Resize((127,127)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    shuffle = True, num_workers = 4)
    return trainloader



batch_size = 28
#如出现“out of memory”的报错信息，可减⼩batch_size或resize
train_iter = load_data_fashion_mnist(batch_size,resize=224)