"""
File: task_a_evaluate_image_to_text_retrieval.py
Authors: Juan A. Rodriguez , Igor Ugarte, Francesc Net, David Serrano
Description:
    - This script is used to evaluate the image to text retrieval system for task a.
    - It uses the test set for retrieval using KNN
    - Quantitative and qualitative results are presented
"""

import os.path
from os import path

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from datasets import Flickr30k, TripletFlickr30kImgToText, TripletFlickr30kTextToImg

from train import fit
from losses import TripletLoss
from models import EmbeddingImageNet, EmbeddingTextNet, TripletImageText, TripletTextImage
import json

cuda = torch.cuda.is_available()


def extract_embeddings(dataloader, model, dim=2, model_id=''):
    model.to('cuda')
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), dim))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()

            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


def main():
    # Load the datasets
    ROOT_PATH = "../../data/"
    TEST_IMG_EMB = ROOT_PATH + "Flickr30k/test_vgg_features.pkl"
    TEST_TEXT_EMB = ROOT_PATH + "Flickr30k/test_fasttext_features.pkl"

    # Method selection
    base = 'ImageToText'
    text_aggregation = 'mean'
    image_features = 'VGG'
    info = 'test'
    model_id = base + '_' + image_features + '_' + text_aggregation + '_textagg_' + "_" + info

    PATH_MODEL = 'models/'

    # Load the test dataset
    test_dataset = Flickr30k(TEST_IMG_EMB, TEST_TEXT_EMB, train=False,
                             text_aggregation=text_aggregation)  # Create the test dataset

    test_dataset_triplet = TripletFlickr30kTextToImg(test_dataset, split='test')

    batch_size = 64

    margin = 1.
    embedding_text_net = EmbeddingTextNet(embedding_size=300, output_size=256, late_fusion=None)
    embedding_image_net = EmbeddingImageNet(output_size=256)
    model = TripletTextImage(embedding_text_net, embedding_image_net, margin=margin)

    # Check if file exists
    if path.exists(PATH_MODEL + model_id + '.pth'):
        print('Loading the model from the disk, {}'.format(model_id + '.pth'))
        checkpoint = torch.load(PATH_MODEL + model_id + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Obtain ground truth from the json file (test.json)
    with open(ROOT_PATH + 'Flickr30k/test.json') as f:
        data = json.load(f)

    gt = {}  # Ground truth as a dictionary with the image filename as key and the list of text id as value
    for item in data:
        gt[item['filename']] = [x['raw'] for x in item['sentences']]

    # Extract embeddings
    print(model)

    #
    # num_retrievals = 1
    # val_embeddings, val_labels = extract_embeddings(test_loader, model, dim, model_id)
    #
    # # Retrieve queries
    # retrieved_imgs = retrieve_imgs(train_embeddings, val_embeddings, k=num_retrievals)
    # retrieval_idx, retrieval_classes = map_idxs_to_targets(retrieved_imgs)
    # labels_test = generate_labels_test()
    #
    # # Compute mAPk
    # map_k = mapk(labels_test, retrieval_classes, k=num_retrievals)
    # print(f'map{num_retrievals}: {map_k}')
    #
    # # Check if the correct label is in the knn
    # new_retrievals = []
    # for idx, retrieval in enumerate(retrieval_classes):
    #     if labels_test[idx] in retrieval:
    #         new_retrievals.append(labels_test[idx])
    #     else:
    #         new_retrievals.append([retrieval[0]])
    #
    # # compute the confusion matrix
    # confusion_matrix = plot_confusion_matrix(labels_test, new_retrievals, show=False)
    #
    # # compute the precision and recall
    # prec, recall = table_precision_recall(confusion_matrix, show=False)
    #
    # print(f'map@k: {map_k}')
    # print(f'precision@k: {np.mean(prec)}')
    # print(f'recall@k: {np.mean(recall)}')
    #
    # # Plot the embeddings
    # image_representation(val_embeddings, labels_test, type='umap')
    #
    # # Qualitative resutls
    # # Plot query-retrieved images
    # query_paths = map_all_query_paths(PATH_TEST)
    # retrieve_paths = map_idxs_to_paths(retrieval_idx, PATH_TRAIN)
    # plot_image_retrievals(query_paths, retrieve_paths, k=num_retrievals)


# Main
if __name__ == '__main__':
    main()
