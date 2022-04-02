import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from operator import truediv
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def apk(actual, predicted, k):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def plot_confusion_matrix(ground_truth, predicted, show=False):
    """
    Plot the confusion matrix. MUST BE MAP1
    Parameters
    ----------
    ground_truth : list
                   A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    # convert ground truth from a list of lists to a list
    ground_truth = [item for sublist in ground_truth for item in sublist]
    # convert predicted from a list of lists to a list
    predicted = [item for sublist in predicted for item in sublist]
    # compute the confusion matrix
    cm = confusion_matrix(ground_truth, predicted)

    if show:
        plt.figure(figsize=(9,9))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = ['forest', 'opencountry', 'tallbuilding', 'mountain', 'street', 'insidecity', 'coast', 'highway']
        plt.xticks(np.arange(8), tick_marks, rotation=45)
        plt.yticks(np.arange(8), tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    return cm


def table_precision_recall(cm, show=False):

    # compute the precision-recall curve
    tp = np.diag(cm)
    prec = list(map(truediv, tp, np.sum(cm, axis=0)))
    rec = list(map(truediv, tp, np.sum(cm, axis=1)))

    # for prec and rec compute the round to 4 decimal places
    prec = [round(x, 4) for x in prec]
    rec = [round(x, 4) for x in rec]
    if show:
        fig, ax = plt.subplots(1, 1)
        data = [prec,
                rec]
        column_labels = ['forest', 'opencountry', 'tallbuilding', 'mountain', 'street', 'insidecity', 'coast', 'highway']
        df = pd.DataFrame(data, columns=column_labels)
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values,
                 colLabels=df.columns,
                 rowLabels=["Precision", "Recall"],
                 rowColours=["yellow"] * 8,
                 colColours=["yellow"] * 8,
                 loc="center",
                 fontSize=100)
        plt.show()

    return prec, rec

def image_representation(features, classes, type='tsne'):
    """
    Plot the image representation.
    Parameters
    ----------
    features : list
               A list of features (order doesn't matter)
    classes : list
              A list of classes (order does matter)
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    color_palette = np.array(sns.color_palette("hls", len(np.unique(classes))))
    color_map = dict(zip(np.unique(classes), color_palette))

    if type == 'tsne':
        # compute the t-SNE image representation
        tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='pca')
        tsne_features = tsne.fit_transform(features)
        #tsne_features = tsne_features.tolist()
        labels = np.unique(classes)

        # for feature, c in zip(tsne_features, labels):
        #     plt.scatter(feature[0], feature[1], c=color_map[c])


        for idx, label in enumerate(labels):
            label_features = [tsne_features[i] for i, x in enumerate(classes) if x == label]
            for feature in label_features:
                plt.scatter(features[0],
                            features[1],
                            c=np.array(color_palette[idx]))
        plt.show()

        # plot the t-SNE image representation
        # plt.figure(figsize=(9,9))
        # plt.scatter(tsne_features[:,0], tsne_features[:,1], c=classes)
        # plt.title("t-SNE image representation")
        # plt.colorbar()
        # tick_marks = ['forest', 'opencountry', 'tallbuilding', 'mountain', 'street', 'insidecity', 'coast', 'highway']
        # plt.xticks(np.arange(8), tick_marks, rotation=45)
        # plt.yticks(np.arange(8), tick_marks)
        # plt.tight_layout()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.show()


def plot_prec_recall_map_k(type=None, **lists_k):
    for model, values in lists_k.items():
        k = np.arange(1, len(values) + 1)
        plt.plot(k, values, label=model, linewidth=2, marker='o')
    if type == 'precision':
        plt.title('Precision in function of k')
    if type == 'recall':
        plt.title('Recall in function of k')
    if type == 'mapk':
        plt.title('mapk in function of k')
    plt.ylim(0, 1)
    plt.xlabel('k')
    plt.ylabel(type)
    plt.legend()
    plt.show()




