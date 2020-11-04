import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
from cnn_utils import *
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

% matplotlib
inline
np.random.seed(1)
torch.manual_seed(1)
batch_size = 24
learning_rate = 0.009
num_epocher = 100

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.


class MyData(Dataset):  # 继承Dataset
    def __init__(self, data, y, transform=None):  # __init__是初始化该类的一些基础参数
        self.transform = transform  # 变换
        self.data = data
        self.y = y

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        return sample, self.y[index]  # 返回该样本


train_dataset = MyData(X_train, Y_train_orig[0],
                       transform=transforms.ToTensor())
test_dataset = MyData(X_test, Y_test_orig[0],
                      transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)