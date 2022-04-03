# todo: plotear precision - recall curve
# todo: plotear representacion del espacio (PCA, TSNE, UMAP)
# todo: plotear mprecision@k, mrecall@k y map@k (solo faltan graficas, lo demas ya esta)

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

# To avoid FAIS crashing (Descomentar los de linux y comentar los de windows)
# import mkl
# mkl.get_max_threads()

PATH_ROOT = '../../data/MIT_split/'
PATH_TRAIN = PATH_ROOT + 'train/'
PATH_TEST = PATH_ROOT + 'test/'
PATH_FEATURES = 'features/'


def compute_features(model_id, model, img_path, train_db):
    """
    Compute the features of an image. The features are computed using the model.
    :param model: the model to use  (e.g. resnet50)
    :param img_path: the path to the image  (e.g. '../../data/MIT_split/train/0/0_0.jpg')
    :param train_db: the database of the training set. It is used to compute the features of the test or train images.
    :return: the features of the image  (numpy array)
    """
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
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ]
        )
        features = []
        classes = []
        for folder in tqdm(os.listdir(PATH), desc='Computing features'):
            folder_path = os.path.join(PATH, folder)
            for image in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image)
                if "base" in model_id:  # case we use just the base model
                    img = cv2.imread(image_path)[:, :, ::-1]
                    img = torch.tensor(img.copy()).permute(2, 0, 1).unsqueeze(0).float()
                    features.append(model(img))
                else:  # case we use siamese or triplet finetuning
                    img = Image.open(image_path)
                    im = transform(img.copy())
                    features.append(model.get_embedding(im.unsqueeze(0)))  # we unsqueeze the image to get a batch of
                    # 1 image
                classes.append(folder)

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


def retrieve_imgs(features_train, features_test, k):
    """
    Retrieve the images from the test set that are similar to the image in the train set.
    :param features_train: the features of the train set
    :param features_test: the features of the test set
    :param k: the number of images to retrieve
    :return: the list of the retrieved images
    """
    # create a faiss index with the features of the train set:
    # Exact Search for L2
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
    retrievals_idx = retrievals.copy()
    for idx_folder, folder in enumerate(os.listdir(PATH_TRAIN)):
        folder_path = os.path.join(PATH_TRAIN, folder)
        # count the elements inside the folder
        for counter_retrievals_class, retrieval in enumerate(retrievals):
            for idx_retrieval, index_of_img_retrieved in enumerate(retrieval):
                if count <= index_of_img_retrieved < count + len(os.listdir(folder_path)):
                    retrievals_idx[counter_retrievals_class][idx_retrieval] = idx_folder
        count += len(os.listdir(folder_path))

    return retrievals, retrievals_idx


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


def map_idxs_to_paths(img_idxs):
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
            for idx_folder, folder in enumerate(os.listdir(PATH_TRAIN)):
                folder_path = os.path.join(PATH_TRAIN, folder)
                if (count <= retrieval_idx) and (retrieval_idx < count + len(os.listdir(folder_path))):
                    img_path_retrievals.append(
                        os.path.join(folder_path, os.listdir(folder_path)[retrieval_idx - count]))
                    break
                count += len(os.listdir(folder_path))
        paths.append(img_path_retrievals)
    return paths


def generate_labels_test():
    """
    Generate the labels of the test set.
    :return: the labels of the test set (numpy array)
    """
    labels = []
    for idx_folder, folder in enumerate(os.listdir(PATH_TRAIN)):
        folder_path = os.path.join(PATH_TEST, folder)
        # generate array of size count of idx_folder number of times
        labels.extend([[idx_folder]] * len(os.listdir(folder_path)))
    return labels


def compute_prec_recall_and_map_for_k(model_id, PATH, model):
    """
    Compute the precision, recall and MAP for k.
    :return: the precision, recall and MAP for k
    """
    with torch.no_grad():
        # Number of iterations of k
        n_iterations = 10
        precision_k = np.zeros(n_iterations)
        reccall_k = np.zeros(n_iterations)
        map_k = np.zeros(n_iterations)

        # Obtain the features of the images: TRAIN
        features_train, _ = compute_features(model_id, model,  img_path=PATH + '/train', train_db=True)
        features_test, _ = compute_features(model_id, model, img_path=PATH + '/train', train_db=False)

        for k in range(n_iterations):
            num_retrievals = k + 1

            # Retrieve the images from the test set that are similar to the image in the train set. retrieve_imgs returns
            # the indexes of the retrieved images, and we map them to the corresponding labels
            _, retrievals = map_idxs_to_targets(retrieve_imgs(features_train, features_test, k=num_retrievals))
            labels_test = generate_labels_test()

            # compute the map@k
            mapk_result = mapk(labels_test, retrievals, k=num_retrievals)

            # new_retrievals is a list of lists, each list contains the indexes of the retrieved images
            # it is created in order to be compatible with the function plot_confusion_matrix
            new_retrievals = []
            for idx, retrieval in enumerate(retrievals):
                if labels_test[idx] in retrieval:
                    new_retrievals.append(labels_test[idx])
                else:
                    new_retrievals.append([retrieval[0]])

            # compute the confusion matrix
            confusion_matrix = plot_confusion_matrix(labels_test, new_retrievals, show=False)

            # compute the precision and recall
            prec, recall = table_precision_recall(confusion_matrix, show=False)

            # append values to the arrays. For prec and recall we use the mean of all classes
            map_k[k] = mapk_result
            # 'How k's it takes in order to get a image of the value of the target class'. Compute prec and rec
            precision_k[k] = np.mean(prec)
            reccall_k[k] = np.mean(recall)

        return map_k, precision_k, reccall_k


# Available backbone models
backbones = {
    '0': 'resnet18',
    '1': 'resnet34',
    '2': 'resnet50',
    '3': 'resnet101',
    '4': 'customCNN',
}

if __name__ == "__main__":
    # PARAMETERS

    # Number of retrievals
    num_retrievals = 1
    # true if you want to plot precision and recall and map in function of num_retrievals
    plot_prec_and_recall_k = True
    # true if you want to save results when plotting precision and recall and map in function of num_retrievals
    saveRes = True

    # Initialize the model. delete pickle file of train and test if you want to recompute the features!
    baseline = 'resnet50'
    method = 'base'
    model_id = baseline + '_' + method

    model = torch.hub.load('pytorch/vision:v0.10.0', baseline, pretrained=True)
    # Remove the last layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)

    with torch.no_grad():
        # Obtain the features of the images: TRAIN
        features_train, classes_train = compute_features(model_id, model, img_path=PATH_TRAIN, train_db=True)
        features_test, classes_test = compute_features(model_id, model, img_path=PATH_TEST, train_db=False)

        # Retrieve the images from the test set that are similar to the image in the train set. retrieve_imgs returns
        # the indexes of the retrieved images, and we map them to the corresponding labels
        retrieved_imgs = retrieve_imgs(features_train, features_test, k=num_retrievals)
        retrieval_idx, retrieval_classes = map_idxs_to_targets(retrieved_imgs)
        labels_test = generate_labels_test()

        # Plot query-retrieved images
        query_paths = map_all_query_paths(PATH_TEST)
        retrieve_paths = map_idxs_to_paths(retrieval_idx)
        plot_image_retrievals(query_paths, retrieve_paths, k=5)

        mapk_result = mapk(labels_test, retrieval_classes, k=num_retrievals)
        print(f'map{num_retrievals}: {mapk_result}')

        if num_retrievals == 1:
            confusion_matrix = plot_confusion_matrix(labels_test, retrieval_classes, show=False)
            prec, recall = table_precision_recall(confusion_matrix, show=False)

        # todo: plotear precision - recall curve

        image_representation(features_test, classes_test, type='umap')

    if plot_prec_and_recall_k:
        map_k, precision_k, recall_k = compute_prec_recall_and_map_for_k(model_id, PATH_ROOT, model)

        plot_prec_recall_map_k(type='precision', Resnet18=precision_k)
        plot_prec_recall_map_k(type='recall', Resnet18=recall_k)
        plot_prec_recall_map_k(type='mapk', Resnet18=map_k)

        # save map_k and precision_k and recall_k in a file
        if saveRes:
            np.save(f'variables/map_k_{model_id}', map_k)
            np.save(f'variables/precision_k_{model_id}', precision_k)
            np.save(f'variables/recall_k_{model_id}', recall_k)

        print(f'map@k: {map_k}')
        print(f'precision@k: {precision_k}')
        print(f'recall@k: {recall_k}')
