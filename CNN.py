import timeimport torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as trans


USE_CUDA = True
def conv5x5(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=5,
                     stride=stride,padding=2,bias=True)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = conv5x5(1,20)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = conv5x5(20,10)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(7*7*10,500)
        self.fc2 = nn.Linear(500,10)

    def forward(self,x):
        # 1x28x28
        o = self.conv1(x)
        o = F.relu(o)
        o = self.pool1(o)
        # 5x14x14
        o = self.conv2(o)
        o = F.relu(o)
        o = self.pool2(o)
        o = o.view(-1,7*7*10)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)


        return o
net = LeNet()
if USE_CUDA:
    net = net.cuda()


print('网络结构如下\n')
print(net)