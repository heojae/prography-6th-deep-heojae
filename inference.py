
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
from matplotlib.pyplot import imshow


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device="cpu"


# i think input is 3 dimension jpg image
# so i remove "transforms.Lambda(lambda x: torch.cat([x, x, x], 0))"

transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                 ])


def image_loader(image_name, transform, device="cpu"):
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image)
    image = image.unsqueeze(0)  #this is for input to model [3,224,224] -> [1,3.224.224]
    return image.to(device)  #i think this could be done in cpu also


def inference(model, image,image_name):
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True).to("cpu").item()

    original_image=Image.open(image_name,'r')
    plt.imshow(np.asarray(original_image))
    plt.title("prediction : "+str(pred))
    plt.show()
    print("prediction :",pred)



# i think inference do not need to be done in gpu, it could be done in cpu
# get model from model.py import VGG16
model = VGG16().to(device)
# get model the best test accuracy.
model.load_state_dict(torch.load("best_mnist_vgg.pt"))

###############################################################################################
# change sample name in here. get jpg is recommended.
image_name="sample1.jpg"
################################################################################################

image = image_loader(image_name, transform=transform, device=device)
inference(model,image,image_name)



























