# 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
import cv2
import torch
import torchvision
from torchvision import transforms


def load_data_fashion_mnist(batch_size, resize=None,root=r'C:\Users\639\maskDectorData\\'):


    trainset = torchvision.datasets.ImageFolder(root, transform=transforms.Compose([
                                                    transforms.Resize((127,127)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    shuffle = True, num_workers = 4)
    return trainloader


def cut_img(imgs, postions, output_dir, img_name):
    length = len(imgs)
    #print(imgs)
    final_img = []
    for i in range(0, length):
        img = cv2.imread(imgs[i])
        position = postions[i]
        crop_img = img[position[2]:position[3], position[0]:position[1]]
        final_img.append(crop_img)
        #print(len(crop_img))
        #print("==" + str(position[0]) + "== " + str(position[1]) + "== " + str(position[2]) + "== " + str(position[3]) + "== ")
        #print(output_dir + (imgs[i].split('\\', 1))[1])
        #cv2.imwrite(output_dir + img_name[i], crop_img)
    return final_img
#batch_size = 28
#如出现“out of memory”的报错信息，可减⼩batch_size或resize
#train_iter = load_data_fashion_mnist(batch_size,resize=224)