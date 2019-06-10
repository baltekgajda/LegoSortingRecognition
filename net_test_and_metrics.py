from datetime import datetime
import torch

import evaluation_metrics


def test_network(net, test_data_loader, device, plots_folder='./results/', plot_name='plot'):
    """Test the network on the test data and calculate metrics

    :param net: trained neural network
    :param test_data_loader: data loader with test data
    :return: dict with accuracy, top 1 error and top 5 error
    """
    predictions = []
    probabilities = []
    true_classes = []
    for inputs, classes in test_data_loader:
        inputs = inputs.to(device)
        true_classes += (classes.tolist())
        batch_probabilities = net(inputs)
        probabilities += batch_probabilities.tolist()
        _, batch_preds = torch.max(batch_probabilities, 1)
        predictions += batch_preds.tolist()

    metrics = {
        'accuracy': evaluation_metrics.accuracy(true_classes, predictions),
        'top_1_error': evaluation_metrics.top_n_error(true_classes, probabilities, n=1),
        'top_5_error': evaluation_metrics.top_n_error(true_classes, probabilities, n=5)
    }
    cumulative_ranks = evaluation_metrics.get_cumulative_ranks(true_classes, probabilities)
    filename = plots_folder + 'cmc_{}_{}'.format(plot_name, datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    evaluation_metrics.plot_cmc(
        cumulative_ranks,
        filename=filename
    )

    return metrics
