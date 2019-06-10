import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from collections import Counter


def accuracy(y_true, y_pred):
    """
    :param y_true: array of correct labels
    :param y_pred: array of predicted labels
    :return: accuracy value
    """
    return accuracy_score(y_true, y_pred, normalize=True)


def top_n_error(y_true, probs, n=1):
    """
    :param y_true: array of correct labels. Labels must be integers
    :param probs: array of arrays. Nested array contain class probabilities for sample
    :return: top_n error
    """

    # This might be slow
    y_true = [(idx, val) for idx, val in enumerate(y_true)]

    # print("y_true: {}".format(y_true))

    # Extracts probability of true class. This might be slow
    y_true = [probs[i][j] for (i, j) in y_true]

    # print("y_true: {}".format(y_true))
    # print("probs before dedup: {}".format(probs))

    # 1. Remove duplicate values of probabilities. 2. Sort desceding. 3. Take first n values
    probs = -np.sort(-np.unique(np.array(probs), axis=1), axis=-1)[:, :n]

    # print("probs: {}".format(probs))

    # Binary array. 1 means that example's class is not in top n predicted classes
    not_in_top_n = [1 if not i in j else 0 for i,j in zip(y_true, probs)]

    # print(not_in_top_n)

    # Sum means how many samples was misclassified. Len means how many samples there are.
    return sum(not_in_top_n) / len(not_in_top_n)


def plot_cmc(ranks, filename='', title='CMC curve'):
    """
    :param ranks: array of ranks. First elem holds probability for rank 1, second for rank 2, etc...
    :param filename: filename where figure will be saved. If not present figure will be displayed
    :param title: title of the figure
    """
    if not ranks:
        raise ValueError('Ranks are not present!')

    x = [i for i in range(1, len(ranks) + 1)]
    y = ranks
    plt.plot(x, y, linestyle=':', marker='.', markersize=10)
    plt.title(title)
    plt.ylim([0, 100])
    plt.xlim([1, None])

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def get_cumulative_ranks(y_true, probs):
    """
    :param y_true: array of correct labels. Labels must be integers
    :param probs: array of arrays. Nested array contain class probabilities for sample
    :return array of samples in rank. First elem holds num of samples of rank 1,
            second elem - num of samples of rank 1 and rank 2, third elem - num of samples of rank 1, rank 2 and rank 3
    """
    ranks = [get_rank(i, j) for i, j in zip(y_true, probs)]
    ranks = Counter(ranks)
    ranks = list(ranks.values())
    return np.cumsum(np.array(ranks)).tolist()


def get_rank(y_true, probs):
    """
    :param y_true: correct label. Must be integer
    :param probs: array of class probabilities for sample
    """
    # Get probability for sample
    y_true = probs[y_true]
    probs = -np.sort(-np.unique(np.array(probs)), axis=-1)
    return probs.tolist().index(y_true) + 1
