import numpy as np
import os.path
from os import path

import numpy as np
import torch
from torchvision.datasets import ImageFolder

from evaluation_metrics import mapk, plot_confusion_matrix, table_precision_recall, image_representation, \
    plot_prec_recall_map_k, plot_image_retrievals
from task_a import retrieve_imgs, map_idxs_to_targets, compute_prec_recall_and_map_for_k, \
    generate_labels_test, compute_features
from model import EmbeddingNet, SiameseNet
from utils import map_all_query_paths, map_idxs_to_paths
from datasets import SiameseMIT_split
import torchvision.transforms as transforms

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

if __name__ == "__main__":
    # Method selection
    backbone = backbones['2']
    method = 'siamese'
    info = ''
    model_id = backbone + '_' + method + info

    PATH_MODEL = 'models/'
    PATH_FEATURES = 'features/'

    # create directory for features if not exists
    if not path.exists(PATH_FEATURES):
        os.makedirs(PATH_FEATURES)

    # Load datasets
    ROOT_PATH = "../../data/"
    TRAIN_IMG_DIR = ROOT_PATH + "MIT_split/train/"
    TEST_IMG_DIR = ROOT_PATH + "MIT_split/test/"

    train_dataset = ImageFolder(TRAIN_IMG_DIR)  # Create the train dataset
    test_dataset = ImageFolder(TEST_IMG_DIR)  # Create the test dataset
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
    train_dataset_siamese = SiameseMIT_split(train_dataset, split='train', transform=transform)
    test_dataset_siamese = SiameseMIT_split(test_dataset, split='test', transform=transform)

    batch_size = 32
    siamese_train_loader = torch.utils.data.DataLoader(train_dataset_siamese, batch_size=batch_size, shuffle=True)
    siamese_test_loader = torch.utils.data.DataLoader(test_dataset_siamese, batch_size=batch_size, shuffle=False)

    # Initialize the model
    embedding_net = EmbeddingNet(backbone, model_id)
    model = SiameseNet(embedding_net)

    model.load_state_dict(torch.load(PATH_MODEL + model_id + '.pth'))

    # Remove the last layer
    print(model)

    # Number of retrievals
    num_retrievals = 1
    # true if you want to plot precision and recall and map in function of num_retrievals
    plot_prec_and_recall_k = True
    # true if you want to save results when plotting precision and recall and map in function of num_retrievals
    saveRes = True
    plot_qualitative = False

    with torch.no_grad():
        # Obtain the features of the images: TRAIN
        features_train, classes_train = compute_features(model_id, model, siamese_train_loader, train_db=True)
        features_test, classes_test = compute_features(model_id, model, siamese_test_loader, train_db=False)
        print(model_id)
        # Retrieve the images from the test set that are similar to the image in the train set. retrieve_imgs
        # returns the indexes of the retrieved images, and we map them to the corresponding labels
        retrieved_imgs = retrieve_imgs(features_train, features_test, k=num_retrievals)
        retrieval_idx, retrieval_classes = map_idxs_to_targets(retrieved_imgs)
        labels_test = generate_labels_test()

        # Plot query-retrieved images
        if plot_qualitative:
            query_paths = map_all_query_paths(PATH_TEST)
            retrieve_paths = map_idxs_to_paths(retrieval_idx, PATH_TRAIN)
            plot_image_retrievals(query_paths, retrieve_paths, k=5)

        mapk_result = mapk(labels_test, retrieval_classes, k=num_retrievals)
        print(f'map{num_retrievals}: {mapk_result}')

        if num_retrievals == 1:
            confusion_matrix = plot_confusion_matrix(labels_test, retrieval_classes, show=False)
            prec, recall = table_precision_recall(confusion_matrix, show=False)

        # todo: plotear precision - recall curve

        image_representation(features_train, classes_train, type='umap')

    if plot_prec_and_recall_k:
        map_k, precision_k, recall_k = compute_prec_recall_and_map_for_k(model_id, PATH_ROOT, model)

        plot_prec_recall_map_k(type='precision', Resnet18=precision_k)
        plot_prec_recall_map_k(type='recall', Resnet18=recall_k)
        plot_prec_recall_map_k(type='mapk', Resnet18=map_k)

        # save map_k and precision_k and recall_k in a file
        if saveRes:
            # create directory if it doesn't exist
            if not os.path.exists(PATH_ROOT + 'results/'):
                os.makedirs(PATH_ROOT + 'results/')
            np.save(f'results/map_k_{model_id}', map_k)
            np.save(f'results/precision_k_{model_id}', precision_k)
            np.save(f'results/recall_k_{model_id}', recall_k)

        print(f'map@k: {map_k}')
        print(f'precision@k: {precision_k}')
        print(f'recall@k: {recall_k}')
