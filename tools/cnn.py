import torch
import torchvision
import torchvision.transforms as transforms

from data.data_pre import readxml
from data.load_data import cut_img

transform = transforms.Compose(
    [transforms.Resize((127,127)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.ToPILImage()])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
   #                                     download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
     #                                     shuffle=True, num_workers=1)
position, val, figure, img_name = readxml()
final_imgs = cut_img(figure, position, "D:\\maskDectorData\\real_img\\", img_name)

final_imgs = transform(final_imgs)
trainset = [final_imgs, val]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=1)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                   download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                 #                        shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3*32*32
from torch.autograd import  Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # Conv2d(in_channels, out_channels, kernel_size,
                                          #        stride = 1, padding=0, dilation=1, groups=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)    # MaxPool2d(kernel_size, stride=None, padding=0,
                                          #           dilation=1, return_indices=False, ceil_mode=False)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   #in_features, out_features, bias = True/False
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   #1、卷积 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))   #2、卷积 -> relu -> pool
        # fully connect
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))                #全卷积后再加relu
        x = F.relu(self.fc2(x))                #全卷积后再加relu
        x = self.fc3(x)                        #全卷积得输出
        return x

# our model
net = Net()   #首先初始化

# Define loss (Cross-Entropy)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()   #交叉熵
# SGD with momentum
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #随机梯度下降

if __name__ ==  '__main__':
    # Train the network
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # warp them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            # loss
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # print statistics
            #running_loss += loss.data[0]
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")
    torch.save(net, 'net.pth')
    # net = torch.load('net.pth')   这里是调用保存好的网络

"""
    print("Beginning Testing")
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
        """