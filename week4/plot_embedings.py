import cv2
import numpy as np
import pickle
import os
import torch
import os.path
from os import path
import faiss
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from evaluation_metrics import mapk, plot_confusion_matrix, table_precision_recall, image_representation, \
    plot_prec_recall_map_k
import matplotlib.pyplot as plt
from evaluation_metrics import mapk, plot_confusion_matrix, table_precision_recall, image_representation, \
    plot_image_retrievals
from PIL import Image
import torchvision.transforms as transforms

from week4.model import EmbeddingNet, SiameseNet

mit_classes = ['0', '1', '2', '3', '4', '5', '6', '7']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
cuda = torch.cuda.is_available()


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(8, 8))
    for i in range(8):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mit_classes)
    plt.show()


def extract_embeddings(dataloader, model):
    model.to('cuda')
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


# Main
if __name__ == '__main__':
    PATH_ROOT = '../../data/MIT_split/'
    PATH_TRAIN = PATH_ROOT + 'train/'
    PATH_TEST = PATH_ROOT + 'test/'
    # Available backbone models
    backbones = {
        '0': 'resnet18',
        '1': 'resnet34',
        '2': 'resnet50',
        '3': 'resnet101',
        '4': 'customCNN',
    }
    # Method selection
    backbone = backbones['2']
    method = 'siamese'
    info = 'fc'
    model_id = backbone + '_' + method + '_' + info

    PATH_MODEL = 'models/'
    PATH_FEATURES = 'features/'

    # create directory for features if not exists
    if not path.exists(PATH_FEATURES):
        os.makedirs(PATH_FEATURES)

    # Load datasets
    ROOT_PATH = "../../data/"
    TRAIN_IMG_DIR = ROOT_PATH + "MIT_split/train/"
    TEST_IMG_DIR = ROOT_PATH + "MIT_split/test/"

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
    train_dataset = ImageFolder(TRAIN_IMG_DIR, transform=transform)  # Create the train dataset
    test_dataset = ImageFolder(TEST_IMG_DIR, transform=transform)  # Create the test dataset

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    embedding_net = EmbeddingNet(backbone)
    model = SiameseNet(embedding_net)

    model.load_state_dict(torch.load(PATH_MODEL + model_id + '.pth'))

    # Remove the last layer
    print(model)
    train_embeddings, train_labels = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings, train_labels)
    val_embeddings, val_labels = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings, val_labels)
