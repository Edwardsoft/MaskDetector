import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms

from data.data_pre import readxml

model = torchvision.models.resnet50(pretrained=True).eval().cuda()

tf = transforms.Compose([

            transforms.Resize(256),

            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]

        )])


with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

position, val, figure = readxml()
src = cv2.imread("D:/images/space_shuttle.jpg") # aeroplane.jpg
image = cv2.resize(src, (224, 224))
image = np.float32(image) / 255.0
image[:,:,] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
image[:,:,] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
image = image.transpose((2, 0, 1))
input_x = torch.from_numpy(image).unsqueeze(0)
print(input_x.size())
pred = model(input_x.cuda())
pred_index = torch.argmax(pred, 1).cpu().detach().numpy()
print(pred_index)
print("current predict class name : %s"%labels[pred_index[0]])
cv2.putText(src, labels[pred_index[0]], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.imshow("input", src)
cv2.waitKey(0)
cv2.destroyAllWindows()