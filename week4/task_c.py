# Task c - Triplet network

import os
import os.path
from os import path

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomHorizontalFlip, RandomRotation

from datasets import SiameseMIT_split, TripletMIT_split
from train import fit
# Mean and std of imagenet dataset
from week4.losses import ContrastiveLoss, TripletLoss
from week4.model import EmbeddingNet, SiameseNet, TripletNet
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
    backbone = backbones['2']
    method = 'triplet'
    model_id = backbone + '_' + method

    train_dataset = ImageFolder(TRAIN_IMG_DIR)  # Create the train dataset
    test_dataset = ImageFolder(TEST_IMG_DIR)  # Create the test dataset

    train_dataset_triplet = TripletMIT_split(train_dataset, split='train', transform=transform)
    test_dataset_triplet = TripletMIT_split(test_dataset, split='test', transform=transform)

    batch_size = 16
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(train_dataset_triplet, batch_size=batch_size, shuffle=True)
    triplet_test_loader = torch.utils.data.DataLoader(test_dataset_triplet, batch_size=batch_size, shuffle=False)

    margin = 1.
    embedding_net = EmbeddingNet(backbone)
    model = TripletNet(embedding_net)

    # Check if file exists
    if path.exists(OUTPUT_MODEL_DIR + model_id + '.pth'):
        print('Loading the model from the disk')
        model.load_state_dict(torch.load(OUTPUT_MODEL_DIR + model_id + '.pth'))

    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 50
    log_interval = 10

    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        model_id)


# main function
if __name__ == "__main__":
    main()
