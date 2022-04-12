import numpy as np
import os
import scipy.io

from adapt_features import select_img_from_feats, select_text_from_feats, generate_negative_idx, project_features

# Load features from db
path_to_db = os.path.join('..', '..', 'data', 'Flickr30k')

# Load features from db
features_text = np.load(os.path.join(path_to_db, 'fasttext_feats.npy'), allow_pickle=True)
features_img = scipy.io.loadmat(os.path.join(path_to_db, 'vgg_feats.mat'))['feats']

# select randomly an image
idx_img, feature_img = select_img_from_feats(features_img)

# select randomly a caption from the image
idx_img, _, feature_text = select_text_from_feats(features_text, idx_img=idx_img)

# select randomly a caption from other image (cannot be the anchor)
idx_img_neg, _, feature_text_neg = select_text_from_feats(features_text,
                                                       idx_img=generate_negative_idx(features_img.shape[0], idx_img))

# project the features into the same space
anchor, positive, negative = project_features(feature_img[:, np.newaxis], feature_text, feature_text_neg)

# todo: compute distance between img and positive, img and negative

print('finished')


