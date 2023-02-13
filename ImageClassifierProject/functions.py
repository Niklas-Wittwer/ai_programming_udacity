#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#PROGRAMMER:Niklas W

# Imports 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL


def load_data(data_dir='flowers'):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Loading the datasets with ImageFolder
    train_image_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
    valid_image_dataset = datasets.ImageFolder(valid_dir, transform = valid_transform)
    test_image_dataset = datasets.ImageFolder(test_dir, transform = test_transform)

    # Defining dataloaders
    trainloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_image_dataset, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_image_dataset, batch_size=64, shuffle = True)
    return trainloader, validloader, testloader, train_image_dataset.class_to_idx

def save_model(model, optimizer, epochs, hidden_layer, class_to_idx, save_dir, dropout=0.3, learnrate=0.001, model_struct="vgg13"):
    '''
    function to save checkpoint for model
    '''
    model.class_to_idx = class_to_idx
    checkpoint = {
        'hidden_layer' : hidden_layer,
        'state_dict' : model.state_dict(),
        'classifier' : model.classifier,
        'class_to_idx' : model.class_to_idx,
        'learnrate' : learnrate,
        'dropout' : dropout,
        'epoch' : epochs,
        'model_structure' : model_struct,
        'optimizer_state_dict' : optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir+"/checkpoint.pth")
               
def load_model(filepath):
    '''
    Function to load model
    '''
    checkpoint = torch.load(filepath)  
    
    model_name = checkpoint["model_structure"]
    if model_name == "resnet18":
        model = models.densenet121(pretrained=True)


    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)


    elif model_name == "vgg13":
        model = models.vgg13(pretrained=True)

        
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learnrate'])
    
    #Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learnrate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with PIL.Image.open(image) as img:
        #Resize and crop PIL image
        img_w, img_h = img.size
        if img_w <= img_h:
            img = img.resize((256, int((256/img_w)*img_h)))
        else:
            img.resize((int((256/img_h)*img_w), 256))
        width, height = img.size
        
        left = (width-224)/2
        right = (width - left)
        upper = (height - 224)/2
        lower = height - upper
        img = img.crop((left, upper, right, lower))

        #Convert PIL Image to numpy array
        np_img = np.array(img)/255
        np_img= (np_img - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        np_img = np_img.transpose(2, 0, 1)
        return np_img
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax