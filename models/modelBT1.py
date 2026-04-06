# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:29:28 2026

@author: tuann
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class moduleDP(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=0):
        super(moduleDP, self).__init__()
        
        self.dw = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,stride=stride,padding=padding,groups=in_channels)
        self.pw = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    def forward(self, x):
        x = F.relu(self.dw(x))
        x = F.relu(self.pw(x))
        return x

class NetBT2(nn.Module):
    def __init__(self, n_class=10):
        super(NetBT2, self).__init__()
        self.conv1 = moduleDP(in_channels=9, out_channels=128, kernel_size=5,stride=1,padding=2)
        self.conv2 = moduleDP(in_channels=128, out_channels=64, kernel_size=3,stride=2,padding=1)
        self.conv3 = moduleDP(in_channels=64, out_channels=128, kernel_size=5,stride=1,padding=2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv4 = moduleDP(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1)
        self.conv5 = moduleDP(in_channels=256, out_channels=128, kernel_size=5,stride=2,padding=0)
        self.conv6 = moduleDP(in_channels=128, out_channels=512, kernel_size=1,stride=1,padding=0)
        self.conv7 = moduleDP(in_channels=512, out_channels=1024, kernel_size=3,stride=2,padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1024, n_class)#cifar-10
        
    def forward(self, x):
        x_gray = T.functional.rgb_to_grayscale(x)
        Gau1 = T.GaussianBlur(kernel_size=3,sigma=(0.5, 0.5))
        Gau2 = T.GaussianBlur(kernel_size=3,sigma=(0.7, 0.7))
        Gau3 = T.GaussianBlur(kernel_size=3,sigma=(0.9, 0.9))
        Gau4 = T.GaussianBlur(kernel_size=3,sigma=(1.1, 1.1))
        Gau5 = T.GaussianBlur(kernel_size=3,sigma=(1.3, 1.3))
        Gau6 = T.GaussianBlur(kernel_size=3,sigma=(1.5, 1.5))
        x_Gau1 = Gau1(x_gray)
        x_Gau2 = Gau2(x_gray)
        x_Gau3 = Gau3(x_gray)
        x_Gau4 = Gau4(x_gray)
        x_Gau5 = Gau5(x_gray)
        x_Gau6 = Gau6(x_gray)
        x = torch.cat((x,x_Gau1,x_Gau2,x_Gau3,x_Gau4,x_Gau5,x_Gau6),dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x
class NetBT1(nn.Module):
    def __init__(self):
        super(NetBT1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=128, kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,stride=1,padding=2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5,stride=2,padding=0)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1,stride=1,padding=0)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,stride=2,padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1024, 10)#cifar-10
    def forward(self, x):
        x_gray = T.functional.rgb_to_grayscale(x)
        Gau1 = T.GaussianBlur(kernel_size=3,sigma=(0.5, 0.5))
        Gau2 = T.GaussianBlur(kernel_size=3,sigma=(0.7, 0.7))
        Gau3 = T.GaussianBlur(kernel_size=3,sigma=(0.9, 0.9))
        Gau4 = T.GaussianBlur(kernel_size=3,sigma=(1.1, 1.1))
        Gau5 = T.GaussianBlur(kernel_size=3,sigma=(1.3, 1.3))
        Gau6 = T.GaussianBlur(kernel_size=3,sigma=(1.5, 1.5))
        x_Gau1 = Gau1(x_gray)
        x_Gau2 = Gau2(x_gray)
        x_Gau3 = Gau3(x_gray)
        x_Gau4 = Gau4(x_gray)
        x_Gau5 = Gau5(x_gray)
        x_Gau6 = Gau6(x_gray)
        x = torch.cat((x,x_Gau1,x_Gau2,x_Gau3,x_Gau4,x_Gau5,x_Gau6),dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x
        
        
        
        