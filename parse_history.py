"""Parse neural network training history and print statistics.

Usage:
  parse_history.py -h | --help
  parse_history.py <json_fname>

Options:
  -h        Show this screen.
  --help    Show this screen.

The history JSON file should have the following structure:
{
    'metadata': {'batch_size': batch_size, 'epochs': num_epochs, 'n_folds': num_folds},
    'results': [
        {
            'fold': fold_index,
            'training': {
                'val_classifier_accuracy': [...],
                'val_decoder_root_mean_squared_error': [...],
                ...
            },
            'evaluation': {
                "classifier_loss": classifier_loss_on_test_set,
                "decoder_loss": decoder_loss_on_test_set,
                "classifier_accuracy": classifier_accuracy_on_test_set,
                "decoder_root_mean_squared_error": decoder_root_mean_squared_error_on_test_set
            },
        },
        ...
    ]

"""

import json
import statistics
import docopt


def test_results(data):
    """
    Extracts test accuracies and RMSEs from the cross-validat results.

    Args:
        data (dict): The dictionary loaded from model-classifier.json
                        (see nnets_stats.md for expected structure).

    Returns:
        tuple: (list of accuracies, list of rmses)
    """
    accuracies = []
    rmses = []
    # Access the list of fold results
    folds = data.get('results', [])

    for fold in folds:
        evaluation = fold.get('evaluation', {})
        acc = evaluation.get('classifier_accuracy', 0.0)
        accuracies.append(acc)

        rmse = evaluation.get('decoder_root_mean_squared_error', 0.0)
        rmses.append(rmse)

    return accuracies, rmses


def print_stats(values):
    """
    Prints the min, median, mean, standard deviation, and max of a list of values.

    Args:
        values (list): A list of numerical values.
    """
    if not values:
        print('No values to compute statistics.')
        return

    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    min_val = min(values)
    max_val = max(values)
    print(
        f'Min: {min_val:.4f}, Median: {median:.4f}, Mean: {mean:.4f}, Std Dev: {stdev:.4f}, Max: {max_val:.4f}'
    )


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    json_fname = args['<json_fname>']
    with open(json_fname, 'r') as f:
        data = json.load(f)
    accuracies, rmses = test_results(data)
    print('Accuracies: ', end='')
    for acc in accuracies:
        print(f'{acc:.4f}', end=', ')
    print()
    print_stats(accuracies)
    print('RMSEs: ', end='')
    for rmse in rmses:
        print(f'{rmse:.4f}', end=', ')
    print()
    print_stats(rmses)
