
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
import argparse
from model import VGG16        # i initialize model in model.py as VGG16
from PIL import Image
from torch.optim.lr_scheduler import StepLR












num_epochs = 5
num_classes = 10
batch_size = 8
learning_rate = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                 ])
# change input size from [1,28,28] -> [1,224,224]
# change 1 dimension to 3 dimension [1,224,224] ->[3,224,224]

################################################################
test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          download=True,
                                          transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

##################################################################


def test( model, device, test_loader):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print('\nTest set- best Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return accuracy



# get model from model.py import VGG16
model = VGG16().to(device)
# get model the best test accuracy.
model.load_state_dict(torch.load("best_mnist_vgg.pt"))



accuracy = test( model, device, test_loader)

#print("test set best accuracy :", accuracy)














