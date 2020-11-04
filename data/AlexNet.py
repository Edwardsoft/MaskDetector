import time
import torch
from torch import nn, optim
import torchvision
import sys
sys.path.append("..")
#import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else
'cpu')
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels,kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使⽤更⼩的卷积窗⼝。除了最后的卷积层外，进⼀步增⼤了输出通道数。
            # 前两个卷积层后不使⽤池化层来减⼩输⼊的⾼和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # 这⾥全连接层的输出个数⽐LeNet中的⼤数倍。使⽤丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。输出改为2，戴了或者没戴
            nn.Linear(4096, 2),
        )

        def forward(self, img):
            feature = self.conv(img)
            output = self.fc(feature.view(img.shape[0], -1))
            return output