from datetime import datetime
import torch
import numpy as np

import evaluation_metrics


def test_network(net, test_data_loader, num_classes, device, plot_name='plot', svm_classifier=None):
    """Test the network on the test data and calculate metrics

    :param net: trained neural network
    :param test_data_loader: data loader with test data
    :param num_classes: number of classes in data set
    :param device: torch device: cuda or cpu
    :param plot_name: name of file where plot will be saved
    :param svm_classifier: svm classification model which follows the network
    :return: dict with accuracy, top 1 error and top 5 error
    """

    with torch.no_grad():
        predictions = []
        probabilities = []
        true_classes = []
        net.to(device)
        for inputs, classes in test_data_loader:
            inputs = inputs.to(device)
            true_classes += (classes.tolist())

            batch_outputs = net(inputs)
            if svm_classifier is not None:
                batch_probabilities = svm_classifier.predict_proba(batch_outputs.numpy())
                batch_probabilities = predict_proba_ordered(batch_probabilities, svm_classifier.classes_,
                                                            np.arange(num_classes))
                probabilities += batch_probabilities.tolist()
                batch_preds = np.argmax(batch_probabilities, 1)
            else:
                probabilities += batch_outputs.tolist()
                _, batch_preds = torch.max(batch_outputs, 1)
            predictions += batch_preds.tolist()

        metrics = {
            'accuracy': evaluation_metrics.accuracy(true_classes, predictions),
            'top_1_error': evaluation_metrics.top_n_error(true_classes, probabilities, n=1),
            'top_5_error': evaluation_metrics.top_n_error(true_classes, probabilities, n=5)
        }
        cumulative_ranks = evaluation_metrics.get_cumulative_ranks(true_classes, probabilities)
        evaluation_metrics.plot_cmc(
            cumulative_ranks,
            filename='./results/cmc_{}_{}'.format(plot_name, datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        )

        return metrics


def predict_proba_ordered(probs, classes_, all_classes):
    """
    extend probability list with classes which did not appears during learning
    :param probs: list of probabilities, output of predict_proba
    :param classes_: clf.classes_
    :param all_classes: all possible classes (superset of classes_)
    :return: probability matrix
    """
    proba_ordered = np.zeros((probs.shape[0], all_classes.size), dtype=np.float)
    sorter = np.argsort(all_classes)  # http://stackoverflow.com/a/32191125/395857
    idx = sorter[np.searchsorted(all_classes, classes_, sorter=sorter)]
    proba_ordered[:, idx] = probs
    return proba_ordered
