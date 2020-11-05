import numpy as np


def sigmoid(Z):
    """
    使用numpy实现sigmoid函数
    Arguments:
    Z -- 任何尺寸的numpy array
    Returns:
    A -- 输出sigmoid(z), 形状和Z一样
    cache -- 就是Z，在反向传播中会用到
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """
    使用numpy实现relu函数
    Arguments:
    Z -- 任何尺寸的numpy array
    Returns:
    A -- 输出relu(z), 形状和Z一样
    cache -- 就是Z，在反向传播中会用到
    """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    实现了relu单元的反向传播
    Arguments:
    dA -- 激活函数的梯度
    cache -- 之前定义的relu函数中的返回值，前向传播之前的Z
    Returns:
    dZ -- 损失函数对Z的梯度
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # relu导数为1(输入大于0)，所以直接直接复制一份dA即可
    dZ[Z <= 0] = 0  # 当输入小于0时，relu导数为0，所以dZ中小于0的数变为0
    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid_backward(dA, cache):
    """
     实现了sigmoid单元的反向传播
    Arguments:
    dA -- 激活函数的梯度
    cache -- 之前定义的sigmoid函数中的返回值，前向传播之前的Z
    Returns:
    dZ -- 损失函数对Z的梯度
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)  # dA乘sigmoid导数
    assert (dZ.shape == Z.shape)
    return dZ


def load_data():
    """
    读取数据
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"])  # 训练样本 shape:(209, 64, 64, 3)
    train_set_y_orig = np.array(train_dataset["train_set_y"])  # 训练样本标签 shape:(1, 209)

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"])  # 测试样本 shape:(50, 64, 64, 3)
    test_set_y_orig = np.array(test_dataset["test_set_y"])  # 测试样本标签 shape:(1, 50)

    classes = np.array(test_dataset["list_classes"][:])  # 标签类别(一共两类：是猫、不是猫)

    train_set_y_orig = train_set_y_orig.reshape((1, -1))  # 确保标签是一行数据 下同
    test_set_y_orig = test_set_y_orig.reshape((1, -1))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
