import cv2
import numpy as np
import glob
import os
import torch
from model import NetSquared
import matplotlib.pyplot as plt

PATH_ROOT = '../../data/MIT_split/test/'

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__=="__main__":
    #Initialize the model
    model = NetSquared()

    # Load the model weights of the week1 checkpoint
    model = load_checkpoint('../week1/NetSquared_checkpoint.pth.tar', model)
    print(model)

    # Remove the last layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)

    with torch.no_grad():
        # Obtain the features of the images
        features = {}
        for folder in os.listdir(PATH_ROOT):
            folder_path = os.path.join(PATH_ROOT, folder)
            for image in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image)
                img = cv2.imread(image_path)[:,:,::-1]
                features = model(torch.tensor(img.copy()).permute(2,0,1).unsqueeze(0).float())



