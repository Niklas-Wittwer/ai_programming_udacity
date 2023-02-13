#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#PROGRAMMER:Niklas W

#imports 
import argparse
from functions import process_image, load_model
import json
import torch
import numpy as np


# Get input args
parser = argparse.ArgumentParser()
# Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
parser.add_argument('--image', type = str, default = 'flowers/test/20/image_04912.jpg', 
                    help = 'path to the image')
parser.add_argument('--checkpoint', type = str, default = 'save_directory/checkpoint.pth',
                    help = 'Path to model checkpoint to use')
parser.add_argument('--gpu', type = str, default = 'cuda',
                    help = 'Use gpu or cpu')
parser.add_argument('--topk', type  = int, default = 5,
                    help = 'Number of top classes to show')
parser.add_argument('--label_map', type  = str, default = 'cat_to_name.json',
                    help = 'path to file that contains label mapping')

input_args = parser.parse_args()

with open(input_args.label_map, 'r') as f:
    cat_to_name = json.load(f)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        device = torch.device("cuda" if input_args.gpu=="gpu" else "cpu")
        model.to(device)
        model.eval()
          
        torch_img = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to("cpu")
        logps = model.forward(torch_img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        
        # Detatch details
        top_p = np.array(top_p.detach())[0] 
        top_class = np.array(top_class.detach())[0]
    
        
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [idx_to_class[label] for label in top_class]
        return top_p, top_labels
    
def main():
    model, optimizer = load_model(input_args.checkpoint)
    top_p, top_labels = predict(input_args.image, model, topk=input_args.topk)
    top_flowers = [cat_to_name[num] for num in top_labels]
    for probability, flower in zip(top_p, top_flowers):
        print("{:25s} {:4.2f}".format(flower, round(float(probability),4)))
if __name__ == "__main__":
    main()