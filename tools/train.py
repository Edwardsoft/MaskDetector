import torch
from torch import nn, optim
from unittest.test import test_loader

from tools.ConvBlock import ResModel

device = 'cuda'


def test():
    model.eval()  # 需要说明是否模型测试
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        img = img.float().to(device)
        label = label.long().to(device)
        out = model(img)  # 前向算法
        loss = criterion(out, label)  # 计算loss
        eval_loss += loss.item() * label.size(0)  # total loss
        _, pred = torch.max(out, 1)  # 预测结果
        num_correct = (pred == label).sum()  # 正确结果
        eval_acc += num_correct.item()  # 正确结果总数

    print('Test Loss:{:.6f},Acc: {:.6f}'
          .format(eval_loss / (len(test_dataset)), eval_acc * 1.0 / (len(test_dataset))))


model = ResModel(6)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

# 开始训练
for epoch in range(num_epocher):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.float().to(device)
        label = label.long().to(device)
        # 前向传播
        out = model(img)
        loss = criterion(out, label)  # loss
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)  # 预测结果
        num_correct = (pred == label).sum()  # 正确结果的数量
        running_acc += num_correct.item()  # 正确结果的总数

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 后向传播计算梯度
        optimizer.step()  # 利用梯度更新W，b参数
    # 打印一个循环后，训练集合上的loss和正确率
    if (epoch + 1) % 1 == 0:
        print('Train{} epoch, Loss: {:.6f},Acc: {:.6f}'.format(epoch + 1, running_loss / (len(train_dataset)),
                                                               running_acc / (len(train_dataset))))
        test()