# Task b - Siamese network

import os
import os.path
from os import path

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomHorizontalFlip, RandomRotation

from datasets import SiameseMIT_split
from train import fit
# Mean and std of imagenet dataset
from week4.losses import ContrastiveLoss
from week4.model import EmbeddingNet, SiameseNet, NetSquared
import wandb

wandb.init(project="M5-week4", entity="celulaeucariota")

# Available backbone models
backbones = {
    '0': 'resnet18',
    '1': 'resnet34',
    '2': 'resnet50',
    '3': 'resnet101',
    '4': 'customCNN',
}


def main():
    # cuda management
    DEVICE = 'cuda'
    cuda = torch.cuda.is_available()

    # Find which device is used
    if cuda and DEVICE == "cuda":
        print(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CAREFUL!! Training the model with CPU')

    # Output directory
    OUTPUT_MODEL_DIR = './models/'

    # Create the output directory if it does not exist
    if not path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    # Load the datasets
    # Transform the output of the Dataset object into Tensor
    transform = transforms.Compose(
        [
            # RandomHorizontalFlip(),
            # RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]
    )
    ROOT_PATH = "../../data/"
    TRAIN_IMG_DIR = ROOT_PATH + "MIT_split/train/"
    TEST_IMG_DIR = ROOT_PATH + "MIT_split/test/"

    # Method selection
    backbone = backbones['4']
    method = 'classification'
    info = ''
    model_id = backbone + '_' + method + info

    train_dataset = ImageFolder(TRAIN_IMG_DIR, transform=transform)  # Create the train dataset
    test_dataset = ImageFolder(TEST_IMG_DIR, transform=transform)  # Create the test dataset

    batch_size = 32
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = NetSquared()

    # Check if file exists
    if path.exists(OUTPUT_MODEL_DIR + model_id + '.pth'):
        print('Loading the model from the disk')
        model.load_state_dict(torch.load(OUTPUT_MODEL_DIR + model_id + '.pth'))

    if cuda:
        model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 50
    log_interval = 10

    wandb.config = {
        "learning_rate": lr,
        "epochs": n_epochs,
        "batch_size": batch_size
    }

    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        model_id)


# main function
if __name__ == "__main__":
    main()
