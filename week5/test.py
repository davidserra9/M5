import pickle

import numpy as np

ROOT_PATH = "../../data/"

TRAIN_IMG_EMB = ROOT_PATH + "Flickr30k/train_vgg_features.pkl"
TEST_IMG_EMB = ROOT_PATH + "Flickr30k/val_vgg_features.pkl"

TRAIN_TEXT_EMB = ROOT_PATH + "Flickr30k/train_bert_features.pkl"
TEST_TEXT_EMB = ROOT_PATH + "Flickr30k/val_bert_features.pkl"


# Load the text embeddings
with open(TRAIN_TEXT_EMB, 'rb') as f:
    text_embeddings = pickle.load(f)

res = text_embeddings.reshape(text_embeddings.shape[0]//5, 5, -1)
print(res.shape)


