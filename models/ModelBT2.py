# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:13:21 2026

@author: tuann
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.SE_Attention import *

class moduleNew(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super(moduleNew, self).__init__()
        self.s = s
        self.Dwd1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                              kernel_size=3, stride=s, padding=1, groups=in_channels, dilation=1)
        self.Dwd1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                              kernel_size=3, stride=s, padding=2, groups=in_channels, dilation=2)
        self.PwOut = nn.Conv2d(in_channels=2*in_channels, out_channels=out_channels, kernel_size=1)
        self.SE = SE(out_channels, 16)
        
    def forward(self, x):
        x_ori = x
        x1 = F.relu(self.Dwd1(x))
        x2 = F.relu(self.Dwd1(x))
        x =  torch.cat((x1,x2),dim=1)
        x = self.shuffle(x)
        x = F.relu(self.PwOut(x))
        x = self.SE(x)
        if self.s==1 and x.size()==x_ori.size():
            x = x_ori + x
        return x
    
    def shuffle(self, x):
        num_group = 2
        b, num_channels, h, w = x.size()
        assert num_channels % num_group == 0
        group_channels = num_channels // num_group
        x = x.reshape(b, num_group, group_channels, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(b, num_channels, h, w)
        return x


class modulePDP(nn.Module):
    def __init__(self, in_channels,out_channels, s):
        super(modulePDP, self).__init__()
        self.Pw1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.Dw = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=s,padding=1,groups=in_channels)
        self.Pw2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.SE = SE(out_channels,16)
        self.PwR = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,stride=s)
        
        self.s = s
    def forward(self, x):
        Pw1 = self.Pw1(x)
        Dw = F.relu(self.Dw(Pw1))
        Pw2 = F.relu(self.Pw2(Dw))
        PDP = self.SE(Pw2)
        if self.s==1 and x.size()==PDP.size():
            FRPDP = x + PDP
        else:
            PwR = F.relu(self.PwR(x))
            #print(PwR.size())
            FRPDP = PwR + PDP
        return F.relu(FRPDP)

class PDPNet(nn.Module):
    def __init__(self, image_size,n_class):
        super(PDPNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.modulePDP_1 = moduleNew(in_channels=32,out_channels=64,s=1)
        self.modulePDP_2 = moduleNew(in_channels=64,out_channels=64,s=1)
        if image_size==32:
            self.modulePDP_3 = moduleNew(in_channels=64,out_channels=128,s=1)
        else:
            self.modulePDP_3 = moduleNew(in_channels=64,out_channels=128,s=2)
        self.modulePDP_4 = moduleNew(in_channels=128,out_channels=128,s=1)
        self.modulePDP_5 = moduleNew(in_channels=128,out_channels=256,s=2)
        self.modulePDP_6 = moduleNew(in_channels=256,out_channels=256,s=1)
        self.modulePDP_7 = moduleNew(in_channels=256,out_channels=256,s=2)
        self.modulePDP_8 = moduleNew(in_channels=256,out_channels=512,s=1)
        self.modulePDP_9 = moduleNew(in_channels=512,out_channels=512,s=2)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1024, n_class)#cifar-10
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.modulePDP_1(x)
        x = self.modulePDP_2(x)
        x = self.modulePDP_3(x)
        x = self.modulePDP_4(x)
        x = self.modulePDP_5(x)
        x = self.modulePDP_6(x)
        x = self.modulePDP_7(x)
        x = self.modulePDP_8(x)
        x = self.modulePDP_9(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
