import torch
#import d2lzh_pytorch as d2l
from data.AlexNet import AlexNet, device
#from data.d2lzh_pytorch import utils
from data.data_pre import readxml
from data.load_data import cut_img

#net = AlexNet()
#print(net)
#lr, num_epochs = 0.001, 5
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer,
#device, num_epochs)

position, val, figure = readxml()
cut_img(figure, position, r"C:\Users\639\maskDectorData\finalImg")