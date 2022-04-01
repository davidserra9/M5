import cv2
import numpy as np
import os
import torch
import os.path
from os import path
import faiss
from sklearn.metrics import average_precision_score
from evaluation_metrics import mapk

PATH_ROOT = '../../data/MIT_split/train/'
PATH_TEST = '../../data/MIT_split/test/'


def compute_features(model, img_path, train_db):
    """
    Compute the features of an image. The features are computed using the model.
    :param model: the model to use  (e.g. resnet50)
    :param img_path: the path to the image  (e.g. '../../data/MIT_split/train/0/0_0.jpg')
    :param train_db: the database of the training set. It is used to compute the features of the test or train images.
    :return: the features of the image  (numpy array)
    """
    # if the file features_resnet_train.npy exists, load it
    if path.exists('features_resnet_train.npy') and train_db:
        features = np.load('features_resnet_train.npy')
    elif path.exists('features_resnet_test.npy') and not train_db:
        features = np.load('features_resnet_test.npy')
    else:
        if train_db:
            PATH = PATH_ROOT
        else:
            PATH = PATH_TEST

        features = []
        for folder in os.listdir(PATH):
            folder_path = os.path.join(PATH, folder)
            for image in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image)
                img = cv2.imread(image_path)[:, :, ::-1]
                img = torch.tensor(img.copy()).permute(2, 0, 1).unsqueeze(0).float()
                features.append(model(img))

        # Transform the features from tensor to numpy array
        features = torch.stack(features)
        features = features.numpy()
        # drop the dimensions equal to 1
        features = np.squeeze(features)

        # Save the features in a file
        if train_db:
            np.save('features_resnet_train.npy', features)
        else:
            np.save('features_resnet_test.npy', features)

    return features


def retrieve_imgs(features_train, features_test, k):
    """
    Retrieve the images from the test set that are similar to the image in the train set.
    :param features_train: the features of the train set
    :param features_test: the features of the test set
    :param k: the number of images to retrieve
    :return: the list of the retrieved images
    """
    # create a faiss index
    index = faiss.IndexFlatL2(features_train.shape[1])
    # add the features of the train set to the index
    index.add(features_train)
    # retrieve the features of the test set
    D, I = index.search(features_test, k)
    # return the retrieved images
    return I


def map_idxs_to_targets(retrievals):
    """
    Convert the retrieved images indexes to the corresponding labels.
    :param retrievals: the list of retrieved images indexes
    :return: the list of the retrieved images labels
    """
    count = 0
    for idx_folder, folder in enumerate(os.listdir(PATH_ROOT)):
        folder_path = os.path.join(PATH_ROOT, folder)
        # count the elements inside the folder
        for counter_retrievals_class, retrieval in enumerate(retrievals):
            for idx_retrieval, index_of_img_retrieved in enumerate(retrieval):
                if count <= index_of_img_retrieved < count + len(os.listdir(folder_path)):
                    retrievals[counter_retrievals_class][idx_retrieval] = idx_folder
        count += len(os.listdir(folder_path))

    return retrievals


def generate_labels_test():
    """
    Generate the labels of the test set.
    :return: the labels of the test set (numpy array)
    """
    labels = []
    for idx_folder, folder in enumerate(os.listdir(PATH_ROOT)):
        folder_path = os.path.join(PATH_TEST, folder)
        # generate array of size count of idx_folder number of times
        labels.extend([idx_folder] * len(os.listdir(folder_path)))
    return labels


if __name__=="__main__":

    # Initialize the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    print(model)

    # Remove the last layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)

    # Number of retrievals
    num_retrievals = 5

    with torch.no_grad():
        # Obtain the features of the images: TRAIN
        features_train = compute_features(model, img_path=PATH_ROOT, train_db=True)
        features_test = compute_features(model, img_path=PATH_TEST, train_db=False)

        # Retrieve the images from the test set that are similar to the image in the train set. retrieve_imgs returns
        # the indexes of the retrieved images and we map them to the corresponding labels
        retrievals = map_idxs_to_targets(retrieve_imgs(features_train, features_test, k=num_retrievals))
        labels_test = generate_labels_test()

        mapk = mapk(labels_test, retrievals, k=num_retrievals)

        print('finished')





