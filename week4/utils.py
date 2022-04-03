import cv2
import numpy as np
import pickle
import os
import torch
import os.path
from os import path
import faiss
from tqdm import tqdm
from evaluation_metrics import mapk, plot_confusion_matrix, table_precision_recall, image_representation, \
    plot_prec_recall_map_k
import matplotlib.pyplot as plt
from evaluation_metrics import mapk, plot_confusion_matrix, table_precision_recall, image_representation, \
    plot_image_retrievals


def map_all_query_paths(test_path):
    """
    Map all the paths of the images in the test set to the corresponding folder.
    :param test_path: the path to the test set
    :return: the list of the paths of the images in the test set
    """
    test_paths = []
    for folder in tqdm(os.listdir(test_path), desc='Mapping paths'):
        folder_path = os.path.join(test_path, folder)
        for image in os.listdir(folder_path):
            test_paths.append(os.path.join(folder_path, image))

    return test_paths


def map_idxs_to_paths(img_idxs, PATH):
    """
    Convert the retrieved images indexes to the corresponding paths.
    :param img_idxs: the list of retrieved images indexes
    :return: the list of the retrieved images paths
    """
    # List of lists containing the paths of the retrieved images
    paths = []

    for retrievals in tqdm(img_idxs, desc='Mapping paths'):
        img_path_retrievals = []
        for retrieval_idx in retrievals:
            count = 0
            for idx_folder, folder in enumerate(os.listdir(PATH)):
                folder_path = os.path.join(PATH, folder)
                if (count <= retrieval_idx) and (retrieval_idx < count + len(os.listdir(folder_path))):
                    img_path_retrievals.append(
                        os.path.join(folder_path, os.listdir(folder_path)[retrieval_idx - count]))
                    break
                count += len(os.listdir(folder_path))
        paths.append(img_path_retrievals)
    return paths


mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          ]
cuda = torch.cuda.is_available()


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
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
