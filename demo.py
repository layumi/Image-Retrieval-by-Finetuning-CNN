# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
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
import scipy.io

plt.ion()   # interactive mode
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='3', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_dir',default='./test_pytorch/gallery',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--query_path', default='./test_pytorch/query/1/1.jpg', type=str, help='test_image_path')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
query_path = opt.query_path
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

# gallery
data_dir = test_dir
image_datasets = datasets.ImageFolder( test_dir, data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32,
                                             shuffle=False, num_workers=4)
#query
query_img = datasets.folder.default_loader(query_path)
query_img = data_transforms(query_img)

use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_best.pth')
    network.load_state_dict(torch.load(save_path))
    return network

#####################################################################
#Show result
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor() 
    images = torch.FloatTensor()
    for data in dataloaders:
        img, label = data
        images = torch.cat((images,img),0)
        #bs, ncrops, c, h, w = inputs.size()
        #inputs = inputs.view(-1, c, h, w)
        n = img.size(0)
        ff = torch.FloatTensor(n,2048).zero_()
        input_img = Variable(img.cuda())
        outputs = model(input_img) 
        ff = outputs.data.cpu()
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)  
    return features, images

def extract_feature_query(model,img):
    c, h, w = img.size()
    img = img.view(-1,c,h,w)
    input_img = Variable(img.cuda())
    outputs = model(input_img)
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff,p=2,dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff

######################################################################
# Load Collected data Trained model
model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
add_block = []
add_block += [nn.Linear(num_ftrs, 30 )]  #number of training classes
model_ft.fc = nn.Sequential(*add_block)
model = load_network(model_ft)

# Load ImageNet Trained model
#model = models.resnet50(pretrained=True)

# remove the final fc layer
model.fc = nn.Sequential()
# change to test modal
model = model.eval()
if use_gpu:
    model = model.cuda()
#Extract feature
gallery_feature,gallery_images = extract_feature(model,dataloaders)
query_feature = extract_feature_query(model,query_img)

print('candidate_images: %d '%gallery_images.size(0))
#######################################################################
# sort the images
def sort_img(qf,gf):
    score = gf*qf
    score = score.sum(1)
    # predict index
    s, index = score.sort(dim=0, descending=True)
    return s,index

s,index = sort_img(query_feature,gallery_feature)

########################################################################
# Visualize the rank result
top10_images = gallery_images[index[0:10]]

print('Top 10 images are as follow:')
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure()
    ax = plt.subplot(3,4,1)
    ax.axis('off')
    imshow(query_img,'query')
    for i in range(10):
        ax = plt.subplot(3,4,i+2)
        ax.axis('off')
        imshow(top10_images[i],'top-%d'%(i+1))
        print(image_datasets[index[i]])
except RuntimeError:
    for i in range(10):
        print(image_datasets.imgs[index[i]])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

