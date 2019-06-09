from __future__ import print_function
from __future__ import division

from sklearn import svm
import torch
import numpy as np
from torch.utils.data import DataLoader
import time

# Setup
feature_extract = True


def train_model_with_svm(model, dataLoaders):
    with torch.no_grad():
        # ugly way to increase batch size
        train_loader = DataLoader(
            dataset=dataLoaders['train'].dataset,
            batch_size=1000,
            num_workers=dataLoaders['train'].num_workers,
        )

        model.classification = None
        device = torch.device("cpu")  # sklearn does not support cuda
        since = time.time()

        model = model.to(device)
        clf = svm.SVC(kernel='linear', probability=True)
        inputs, labels = iter(train_loader).next()
        # fit svm
        outputs = model(inputs)
        clf.fit(outputs.numpy(), labels.data.numpy())

        # evaluate classifier on test set
        predictions = clf.predict(outputs.numpy())
        corrects = np.ndarray.sum(predictions == labels.data.numpy())
        print('SVM train Acc: {:.4f}'.format(corrects / train_loader.batch_size))

        print(labels.data.numpy())
        # evaluate classifier on validation set
        running_corrects = 0
        for inputs, labels in dataLoaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predictions = clf.predict(outputs.numpy())
            running_corrects += np.ndarray.sum(predictions == labels.data.numpy())

        print('SVM val Acc: {:.4f}'.format(running_corrects / len(dataLoaders['val'].sampler)))

        time_elapsed = time.time() - since
        print('SVM training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return clf