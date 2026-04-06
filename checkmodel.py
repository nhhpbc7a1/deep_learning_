# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022

@author: tuann
"""
from torchsummary import summary
from models.ModelBT2 import *

#model = moduleNew(in_channels, out_channels, s)

model = PDPNet(image_size=224,n_class=100)

model = model.cuda()
print ("model")
print (model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
#print(model)
#model.cuda()
summary(model, (3, 224, 224))
#summary(model, (3, 32, 32))