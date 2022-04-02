import matplotlib.pyplot as plt
import numpy as np
from evaluation_metrics import plot_prec_recall_map_k

map_resnet18_L2 = np.load('variables/map_k_resnet18.npy')
map_resnet50_L2 = np.load('variables/map_k_resnet50.npy')
map_resnet101_L2 = np.load('variables/map_k_resnet101.npy')

prec_resnet18_L2 = np.load('variables/precision_k_resnet18.npy')
prec_resnet50_L2 = np.load('variables/precision_k_resnet50.npy')
prec_resnet101_L2 = np.load('variables/precision_k_resnet101.npy')

recall_resnet18_L2 = np.load('variables/recall_k_resnet18.npy')
recall_resnet50_L2 = np.load('variables/recall_k_resnet50.npy')
recall_resnet101_L2 = np.load('variables/recall_k_resnet101.npy')

plot_prec_recall_map_k(type='mapk', resnet18_L2=map_resnet18_L2, resnet50_L2=map_resnet50_L2, resnet101_L2=map_resnet101_L2)
plot_prec_recall_map_k(type='precision', resnet18_L2=prec_resnet18_L2, resnet50_L2=prec_resnet50_L2, resnet101_L2=prec_resnet101_L2)
plot_prec_recall_map_k(type='recall',  resnet18_L2=recall_resnet18_L2,  resnet50_L2=recall_resnet50_L2, resnet101_L2=recall_resnet101_L2)

