# 内容参考 https://blog.csdn.net/chenkz123/article/details/79640658
# 内容参考https://blog.csdn.net/weixin_43615222/article/details/84577293
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import h5py
import scipy


# 导入必要的包
def get_files(file_dir):
    male = []
    label_male = []
    female = []
    label_female = []

    for file in os.listdir(file_dir + '/male'):
        male.append(file_dir + '/male' + '/' + file)
        label_male.append(1)  # 添加标签 这里用1 0 代表男女
    for file in os.listdir(file_dir + '/female'):
        female.append(file_dir + '/female' + '/' + file)
        label_female.append(0)

    # 把所有数据集进行合并
    image_list = np.hstack((male, female))
    label_list = np.hstack((label_male, label_female))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list
    # 返回两个list 分别为图片文件名及其标签  顺序已被打乱


dataset = '.\dataset'
image_list, label_list = get_files(dataset)
##################################################测试
print(len(image_list))
print(len(label_list))

# np.random.rand(n, x, y, 3) 其中n为图片数目，x y为图片像素，注意640*480即y*x
# 将图片数据暂时存入Train_image Train_label Test_image Test_label
Train_image = np.random.rand(len(image_list) - 10, 480, 640, 3).astype('float32')
Train_label = np.random.rand(len(image_list) - 10, 1).astype('float32')

Test_image = np.random.rand(10, 480, 640, 3).astype('float32')
Test_label = np.random.rand(10, 1).astype('float32')
for i in range(len(image_list) - 10):
    Train_image[i] = np.array(plt.imread(image_list[i]))
    Train_label[i] = np.array(label_list[i])

for i in range(len(image_list) - 10, len(image_list)):
    Test_image[i + 10 - len(image_list)] = np.array(plt.imread(image_list[i]))
    Test_label[i + 10 - len(image_list)] = np.array(label_list[i])

# 写入data.h5文件

f = h5py.File('datadata.h5', 'w')
f.create_dataset('X_train', data=Train_image)
f.create_dataset('y_train', data=Train_label)
f.create_dataset('X_test', data=Test_image)
f.create_dataset('y_test', data=Test_label)
f.close()
train_dataset = h5py.File('datadata.h5', 'r')
train_set_x_orig = np.array(train_dataset['X_train'][:])  # your train set features
train_set_y_orig = np.array(train_dataset['y_train'][:])  # your train set labels
test_set_x_orig = np.array(train_dataset['X_test'][:])  # your train set features
test_set_y_orig = np.array(train_dataset['y_test'][:])  # your train set labels
f.close()

# 读写测试
print(train_set_x_orig.shape)
print(train_set_y_orig.shape)

print(train_set_x_orig.max())
print(train_set_x_orig.min())

print(test_set_x_orig.shape)
print(test_set_y_orig.shape)

####图片读取  注意/255，否则无法读取
plt.imshow(train_set_x_orig[2] / 255)
print(train_set_y_orig[2])
plt.show()