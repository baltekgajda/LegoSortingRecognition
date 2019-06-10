from __future__ import print_function
from __future__ import division

import time

import numpy as np
from sklearn import svm
from sklearn.svm.base import BaseSVC
import torch
from torch.utils.data import DataLoader

from net_test_and_metrics import calculate_net_output


def train_model_with_svm(model, data_loaders, kernel='l'):
    """Train svm classifier

    :param model: trained neural network which is an input for svm
    :param data_loaders: data loaders with training and validation set
    :param kernel: svm kernel type: 'l' (linear), 'q' (quadratic), 'e' (exponential)
    :return: trained sklearn.svm classifier
    """
    with torch.no_grad():
        # ugly way to increase batch size, cause svc does not support online learning
        # consider using SGDClassifier whith Kernel Approximation which supports online learning

        model.classification = None  # remove fully connected layer
        device = torch.device("cpu")  # sklearn does not support cuda
        since = time.time()

        model = model.to(device)
        outputs, true_labels = calculate_net_output(model, data_loaders['train'], device)

        clf = _build_svm_classifier(kernel=kernel)
        # fit svm
        clf.fit(outputs, true_labels)

        # evaluate classifier on test set
        # predictions = clf.predict(outputs)
        # corrects = np.ndarray.sum(np.ndarray(predictions) == np.ndarray(true_labels))
        # print('SVM train Acc: {:.4f}'.format(corrects / len(true_labels)))

        # evaluate classifier on validation set
        running_corrects = 0
        for inputs, labels in data_loaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predictions = clf.predict(outputs.numpy())
            running_corrects += np.ndarray.sum(predictions == labels.data.numpy())

        print('SVM val Acc: {:.4f}'.format(running_corrects / len(data_loaders['val'].sampler)))

        time_elapsed = time.time() - since
        print('SVM training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return clf


def _build_svm_classifier(kernel: str) -> BaseSVC:
    if kernel == 'l':
        return svm.SVC(kernel='linear', probability=True)
    if kernel == 'q':
        return svm.SVC(kernel='poly', degree=2, probability=True)
    if kernel == 'e':
        return svm.SVC(kernel='rbf', probability=True)
    else:
        raise ("Unsupported kernel type. Expected 'l' (linear), 'q' (quadratic), 'e' (exponential), got: ", kernel)
