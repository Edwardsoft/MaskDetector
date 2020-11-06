# 从文件加载数据的函数
import os

import cv2
import h5py
import numpy as np
from xml.dom.minidom import parse

def load_data():
    # 把文件读取到内存中
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])  # 获取训练集特征
    train_y_orig = np.array(train_dataset["train_set_y"][:])  # 获取训练标签

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])  # 获取测试集特征
    test_y_orig = np.array(test_dataset["test_set_y"][:])  # 获取测试标签

    classes = np.array(test_dataset["list_classes"][:])  # 类别，即 1 和 0

    # 现在的数据维度是 (m,)，我们要把它变成 (1, m)，m 代表样本数量
    train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))

    return train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes

#处理数据
def data_progre(url):
    figure = []  # 图像路径
    position = []  # 人脸位置
    val = []  # 标签
    img_realname = []  # 图像真名
    train_url = url + r"\train"
    for root, dirs, files in os.walk(train_url):
        for file in files:
            if os.path.splitext(file)[1] == ".xml":
                domTree = parse(train_url + "\\" + file)
                # 文档根元素
                rootNode = domTree.documentElement
                # 所有顾客
                peoples = rootNode.getElementsByTagName("object")
                i = 1
                for people in peoples:
                    figure.append(train_url + "\\" + os.path.splitext(file)[0] + ".jpg")
                    img_realname.append(os.path.splitext(file)[0] + "_" + str(i) + ".jpg")
                    i = i + 1
                    type = people.getElementsByTagName('name')[0].firstChild.data
                    if type == "face":
                        val.append(0)
                    else:
                        val.append(1)
                    bd = people.getElementsByTagName('bndbox')[0]
                    print(file)
                    pos = []
                    xmin = int(bd.getElementsByTagName('xmin')[0].firstChild.data)
                    ymin = int(bd.getElementsByTagName('ymin')[0].firstChild.data)
                    xmax = int(bd.getElementsByTagName('xmax')[0].firstChild.data)
                    ymax = int(bd.getElementsByTagName('ymax')[0].firstChild.data)
                    pos.append(xmin)
                    pos.append(xmax)
                    pos.append(ymin)
                    pos.append(ymax)
                    position.append(pos)
    print(len(position))
    print(len(val))
    print(len(figure))
    return position, val, figure, img_realname

def cut_img(imgs, postions, output_dir, img_name):
    length = len(imgs)
    print(imgs)
    for i in range(0, length):
        print(imgs[i])
        img = cv2.imread(imgs[i])
        position = postions[i]
        crop_img = img[position[2]:position[3], position[0]:position[1]]
        #print(len(crop_img))
        #print("==" + str(position[0]) + "== " + str(position[1]) + "== " + str(position[2]) + "== " + str(position[3]) + "== ")
        print(output_dir + (imgs[i].split('\\', 1))[1])
        cv2.imwrite(output_dir + img_name[i], crop_img)


position, val, figure, img_name = data_progre("D:\\maskDectorData")
cut_img(figure, position, "D:\\maskDectorData\\train\\real_img\\", img_name)
