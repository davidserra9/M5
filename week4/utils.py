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
from PIL import Image
import torchvision.transforms as transforms

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




PATH_ROOT = '../../data/MIT_split/'
PATH_TRAIN = PATH_ROOT + 'train/'
PATH_TEST = PATH_ROOT + 'test/'
PATH_FEATURES = 'features/'

def compute_features(model_id, model, dataloader, train_db):
    """
    Compute the features of an image. The features are computed using the model.
    :param model: the model to use  (e.g. resnet50)
    :param img_path: the path to the image  (e.g. '../../data/MIT_split/train/0/0_0.jpg')
    :param train_db: the database of the training set. It is used to compute the features of the test or train images.
    :return: the features of the image  (numpy array)
    """
    print(model_id)
    # if the file features_resnet_train.npy exists, load it
    if path.exists(PATH_FEATURES + model_id + '_train.pkl') and train_db:
        with open(PATH_FEATURES + model_id + '_train.pkl', 'rb') as f:
            features_and_classes = pickle.load(f)
        features = features_and_classes['features']
        classes = features_and_classes['classes']

    elif path.exists(PATH_FEATURES + model_id + '_test.pkl') and not train_db:
        with open(PATH_FEATURES + model_id + '_test.pkl', 'rb') as f:
            features_and_classes = pickle.load(f)
        features = features_and_classes['features']
        classes = features_and_classes['classes']

    else:
        if train_db:
            PATH = PATH_TRAIN
        else:
            PATH = PATH_TEST

        model.to('cuda')
        with torch.no_grad():
            model.eval()
            embeddings = np.zeros((len(dataloader.dataset), 2048))
            labels = np.zeros(len(dataloader.dataset))
            k = 0
            for images, target in dataloader:
                if cuda:
                    images = images.cuda()
                embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
                labels[k:k + len(images)] = target.numpy()
                k += len(images)
        print(embeddings)

        # features = []
        # classes = []
        # for folder in tqdm(os.listdir(PATH), desc='Computing features'):
        #     folder_path = os.path.join(PATH, folder)
        #     for image in os.listdir(folder_path):
        #         image_path = os.path.join(folder_path, image)
        #         if "base" in model_id:  # case we use just the base model
        #             img = cv2.imread(image_path)[:, :, ::-1]
        #             img = torch.tensor(img.copy()).permute(2, 0, 1).unsqueeze(0).float()
        #             features.append(model(img))
        #         else:  # case we use siamese or triplet finetuning
        #             img = Image.open(image_path)
        #             im = transform(img.copy())
        #             features.append(model.get_embedding(im.unsqueeze(0)))  # we unsqueeze the image to get a batch of
        #             # 1 image
        #         classes.append(folder)

        # Transform the features from tensor to numpy array
        features = torch.stack(features)
        features = features.numpy()
        # drop the dimensions equal to 1
        features = np.squeeze(features)

        features_and_classes = {'features': features, 'classes': classes}
        # Save the features in a file
        if train_db:
            with open(PATH_FEATURES + model_id + '_train.pkl', 'wb') as handle:
                pickle.dump(features_and_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(PATH_FEATURES + model_id + '_test.pkl', 'wb') as handle:
                pickle.dump(features_and_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return features, classes