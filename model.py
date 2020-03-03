import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F

# i implement VGG16 layer in here.
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #######################################################################################
        # i think i need to torch.cat before to first Dense layer and
        # i need to change [1,64,112,112] -> [1,512,7,7]
        # so i implement skip1 for this.
        self.skip1 = nn.Conv2d(64,512,kernel_size=6,padding=1,stride=18)
        #######################################################################################
        # i think dilation makes more efficiently to here, but i do not need to do this
        # so i choose skip1 for simple not for efficiently
        #self.skip2 = nn.Conv2d(64,512,kernel_size=6,padding=1,stride=1, dilation=3)


        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

        # fully conected layers:
        self.fc6 = nn.Linear(7*7*512*2, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 10) # i change 1000 ->10

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, training=True):

        x = F.relu(self.batchnorm1(self.conv1_1(x)))
        x = F.relu(self.batchnorm1(self.conv1_2(x)))
        x = self.pool(x)

        input_conv2_1 = x
        #print(input_conv2_1.size())
        #################################################################################
        skip_connection = self.skip1(x)
        #print("skip_connection",skip_connection.size())
        #################################################################################
        x = F.relu(self.batchnorm2(self.conv2_1(x)))
        x = F.relu(self.batchnorm2(self.conv2_2(x)))
        x = self.pool(x)

        x = F.relu(self.batchnorm3(self.conv3_1(x)))
        x = F.relu(self.batchnorm3(self.conv3_2(x)))
        x = F.relu(self.batchnorm3(self.conv3_3(x)))
        x = self.pool(x)

        x = F.relu(self.batchnorm4(self.conv4_1(x)))
        x = F.relu(self.batchnorm4(self.conv4_2(x)))
        x = F.relu(self.batchnorm4(self.conv4_3(x)))
        x = self.pool(x)


        x = F.relu(self.batchnorm5(self.conv5_1(x)))
        x = F.relu(self.batchnorm5(self.conv5_2(x)))
        x = F.relu(self.batchnorm5(self.conv5_3(x)))
        x = self.pool(x)
        #print("x.szie()",x.size())

        ###################################################################################################
        # i implement skip connection here
        #x= x+ skip_connection
        # i think torch.cat could make better result than just adding.
        x = torch.cat((x, skip_connection), dim=1)
        ####################################################################################################
        # x = x.view(-1, 7 * 7 * 512)
        # i change dimmension for torch.cat()
        x = x.view(-1, 7 * 7 * 512*2)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        return x