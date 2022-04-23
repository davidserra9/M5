import pickle

import numpy as np

ROOT_PATH = "../../data/"

TRAIN_IMG_EMB = ROOT_PATH + "Flickr30k/train_FasterRCNN_features.pkl"
TEST_IMG_EMB = ROOT_PATH + "Flickr30k/val_FasterRCNN_features.pkl"

TRAIN_TEXT_EMB = ROOT_PATH + "Flickr30k/train_bert_features.pkl"
TEST_TEXT_EMB = ROOT_PATH + "Flickr30k/val_bert_features.pkl"

for split in ["train", "val"]:
    # Load the text embeddings
    with open(ROOT_PATH + "Flickr30k/"+split+"_FasterRCNN_features.pkl", 'rb') as f:
        im_emb = pickle.load(f)

    res = np.transpose(im_emb)
    print("out size", res.shape)

    with open(ROOT_PATH + "Flickr30k/"+split+"_FasterRCNN_features.pkl", 'wb') as f:
        pickle.dump(res, f)


