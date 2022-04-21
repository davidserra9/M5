import json
import os.path
import pickle
from os import path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from datasets import Flickr30k
from models import EmbeddingImageNet, EmbeddingTextNet, TripletImageText
from week4.evaluation_metrics import mapk, plot_confusion_matrix, table_precision_recall

image_labels = [i for i in range(1, 1000 + 1)]

# Qualitative results
num_samples = 10
# Create random samples
random_samples = np.random.choice(image_labels, num_samples, replace=False)

for sample in random_samples:
    print(sample)
    # Get image embedding from batch
print()

