#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#PROGRAMMER:Niklas W

#Imports
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_layer, output_layer_size, hidden_layers, dropout=0.3):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_layer, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(x, y) for x,y in zip(hidden_layers[:-1], hidden_layers[1:])])
        self.output = nn.Linear(hidden_layers[-1], output_layer_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        output = F.log_softmax(self.output(x), dim=1)
        return output

def create_model(model_name='vgg13', output_size=102, device="cuda", hidden_units=[1024], learnrate=0.001):
    '''
    Function to create a model and classifier for training. Returns model and optimizer
    '''
    model_names = ['resnet18', 'alexnet', 'vgg13']
    if model_name not in model_names:
        print("model architecture must be of resnet18, alexnet or vgg13")
    else:
        if model_name == "resnet18":
            model = models.densenet121(pretrained=True)
            input_size = 1024
            
        elif model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            input_size = 9216
            
        elif model_name == "vgg13":
            model = models.vgg13(pretrained=True)
            input_size = 25088
        #Freeze parameters 
        for param in model.parameters():
            param.requires_grad = False
    
        device = torch.device(device)
        model.classifier = Network(input_size, output_size, hidden_units)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
        model.to(device)
        return model, optimizer
        