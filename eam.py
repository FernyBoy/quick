# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entropic Associative Memory Experiments

Usage:
  eam -h | --help
  eam (-n | -f) [--domain=DOMAIN] [--runpath=PATH ] [ -l (en | es) ]
  eam (-e <experiment> | -r) [--num-classes=NUM] [--domain=DOMAIN] [--runpath=PATH ] [ -l (en | es) ]

Options:
  -h    Show this screen.
  -n    Trains the neural network (classifier+autoencoder).
  -f    Generates Features for all data using the encoder.
  -e    Run the experiment (options 1 or 2).
  -r    Generate images from testing data and memories of them.
  --num-classes=NUM   Number of classes to use. Defaults to value of constants.n_labels.
  --domain=DOMAIN   Size of memory (columns). Defaults to value of constants.domain
  --runpath=PATH   Path to directory where everything will be saved [default: runs]
  -l        Chooses Language for graphs.
"""

from associative import AssociativeMemory
import constants
import dataset
import neural_net
import typing
import seaborn
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import gettext
import gc
from docopt import docopt
import png
import sys
import os
import qudeq

sys.setrecursionlimit(10000)


# A trick to avoid getting a lot of errors at edition time because of
# undefined '_' gettext function.
if typing.TYPE_CHECKING:

    def _(message):
        pass


# Translation
gettext.install('eam', localedir=None, names=None)


# region Plotting functions -------------------------------------------------------------


def plot_metrics_graph(
    pre_mean,
    rec_mean,
    acc_mean,
    ent_mean,
    pre_std,
    rec_std,
    acc_std,
    xlabels,
    suffix,
    es,
    xtitle=None,
    ytitle=None,
    sub_dir=None,
):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))

    full_length = 100.0
    step = 0.1
    main_step = full_length / len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step

    # Gives space to fully show markers in the top.
    ymax = full_length + 2

    # Replace undefined precision with 1.0.
    pre_mean = np.nan_to_num(pre_mean, copy=False, nan=100.0)

    plt.errorbar(x, acc_mean, fmt='b-s', yerr=acc_std, label=_('Accuracy'))
    plt.errorbar(x, pre_mean, fmt='r--o', yerr=pre_std, label=_('Precision'))
    plt.errorbar(x, rec_mean, fmt='g:^', yerr=rec_std, label=_('Recall'))
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)

    if xtitle is None:
        xtitle = _('Range Quantization Levels')
    if ytitle is None:
        ytitle = _('Percentage')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors', ['cyan', 'purple'])
    Z = [[0, 0], [0, 0]]
    levels = np.arange(0.0, xmax, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    cbar = plt.colorbar(CS3, orientation='horizontal')
    cbar.set_ticks(x)
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label(_('Entropy'))

    graph_name = constants.graph_name(es) + suffix + '-metrics' + _('-english')
    graph_filename = constants.figure_filename(graph_name, es, sub_dir=sub_dir)
    plt.savefig(graph_filename, dpi=600)


def plot_responses_graph(
    mean_behaviours,
    stdv_behaviours,
    x_labels,
    suffix,
    es,
    xtitle=None,
    ytitle=None,
    sub_dir=None,
):
    response_idxs = [idx for idx in constants.response_behaviours]
    response_labels = [constants.response_labels[idx] for idx in response_idxs]
    response_colors = [constants.response_colors[idx] for idx in response_idxs]
    # Rows are memory sizes, and columns are behaviours. We select only the
    # response behaviors, and normalize them to percentage of responses.
    means = mean_behaviours[:, response_idxs]
    stdvs = stdv_behaviours[:, response_idxs]
    # We transpose behaviours, as we want to plot them as stacked bars.
    means = means.transpose()
    stdvs = stdvs.transpose()
    # Division along the columns, to get percentage of responses for each behavior.
    means = 100.0 * means / np.sum(means, axis=0)
    stdvs = 100.0 * stdvs / np.sum(means, axis=0)

    plt.clf()
    full_length = 100.0
    step = 0.1
    main_step = full_length / len(x_labels)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5  # the width of the bars: can also be len(x) sequence

    cum = np.zeros(len(constants.memory_sizes), dtype=float)
    for values, errors, color, label in zip(
        means, stdvs, response_colors, response_labels
    ):
        total = np.sum(values)
        if total > 0:
            plt.bar(
                x,
                values,
                width,
                bottom=cum,
                label=label,
                yerr=errors,
                color=color,
            )
            cum += values

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, x_labels)

    if xtitle is None:
        xtitle = _('Memory size (rows)')
    if ytitle is None:
        ytitle = _('Percentage')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.legend(loc=0)
    plt.grid(axis='y')

    graph_name = constants.graph_name(es) + '-responses' + suffix + _('-english')
    graph_filename = constants.figure_filename(graph_name, es, sub_dir=sub_dir)
    plt.savefig(graph_filename, dpi=600)


def plot_conf_matrix(matrix, xtags, ytags, name, es, sub_dir=None):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(
        matrix,
        xticklabels=xtags,
        yticklabels=ytags,
        vmin=0.0,
        vmax=1.0,
        annot=False,
        cmap='Blues',
    )
    plt.xlabel(_('Prediction'))
    plt.ylabel(_('Label'))
    filename = constants.figure_filename(name, es, sub_dir)
    plt.savefig(filename, dpi=600)


def plot_memory(memory: AssociativeMemory, name, es, fold, sub_dir=None):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    seaborn.heatmap(
        memory.relation / memory.max_value,
        vmin=0.0,
        vmax=1.0,
        annot=False,
        cmap='coolwarm',
    )
    plt.xlabel(_('Characteristics'))
    plt.ylabel(_('Values'))
    filename = constants.figure_filename(name, es, fold, sub_dir)
    plt.savefig(filename, dpi=600)


# endregion Plotting functions ----------------------------------------------------

# region Auxiliary functions -------------------------------------------------------------


def filter_by_labels(
    filling_features, filling_labels, testing_features, testing_labels, threshold, es
):
    mask = testing_labels < constants.memory_labels
    testing_labels = testing_labels[mask]
    testing_features = testing_features[mask]
    mask = filling_labels < threshold
    filling_labels = filling_labels[mask]
    filling_features = filling_features[mask]
    return filling_features, filling_labels, testing_features, testing_labels


def load_features_and_labels(threshold, es, fold):
    suffix = constants.filling_suffix
    filling_features_filename = constants.features_name(es) + suffix
    filling_features_filename = constants.shared_data_filename(
        filling_features_filename, fold
    )
    filling_labels_filename = constants.labels_name(es) + suffix
    filling_labels_filename = constants.shared_data_filename(
        filling_labels_filename, fold
    )

    suffix = constants.testing_suffix
    testing_features_filename = constants.features_name(es) + suffix
    testing_features_filename = constants.shared_data_filename(
        testing_features_filename, fold
    )
    testing_labels_filename = constants.labels_name(es) + suffix
    testing_labels_filename = constants.shared_data_filename(
        testing_labels_filename, fold
    )

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    # Reduces the original data to only the classes for the current experiment,
    # given the number of labels.
    filling_features, filling_labels, testing_features, testing_labels = (
        filter_by_labels(
            filling_features,
            filling_labels,
            testing_features,
            testing_labels,
            threshold,
            es,
        )
    )
    return filling_features, filling_labels, testing_features, testing_labels


# endregion Auxiliary functions ----------------------------------------------------------

# region Neural network construction functions -------------------------------------------


def save_history(history, prefix, es):
    """Saves the stats of neural networks.

    Neural networks stats may come either as a History object, that includes
    a History.history dictionary with stats, or directly as a dictionary.
    """
    stats = {}
    stats['history'] = []
    for h in history:
        while not ((type(h) is dict) or (type(h) is list)):
            h = h.history
        stats['history'].append(h)
    with open(constants.json_filename(prefix, es), 'w') as outfile:
        json.dump(stats, outfile)


def save_conf_matrix(matrix, prefix):
    name = prefix + constants.matrix_suffix
    plot_conf_matrix(matrix, range(matrix.shape[0]), range(matrix.shape[1]), name, None)
    filename = constants.data_filename(name)
    np.save(filename, matrix)


# endregion Neural network results -------------------------------------------------------

# region Memory evaluation functions -----------------------------------------------------


def optimum_indexes(precisions, accuracies):
    """Finds the indexes of the best n memory sizes according to precision and accuracy."""
    accs = []
    for idx, acc in enumerate(accuracies):
        accs.append((acc, idx))
    accs.sort(reverse=True, key=lambda tuple: tuple[0])
    return [t[1] for t in accs[: constants.n_best_memory_sizes]]


def save_learned_params(mem_sizes, fill_percents, es):
    """Saves the best memory sizes and filling percentages found.

    The parameters are saved as a 2-row numpy array, where the first row
    contains the memory sizes, and the second row contains the filling
    percentages.
    """
    name = constants.learn_params_name(es)
    filename = constants.data_filename(name, es, None)
    np.save(filename, np.array([mem_sizes, fill_percents], dtype=int))


def load_learned_params(es):
    name = constants.learn_params_name(es)
    filename = constants.data_filename(name, es)
    params = np.load(filename)
    size_fill = [(params[0, j], params[1, j]) for j in range(params.shape[1])]
    return size_fill


def calculate_metrics(behaviour, es):
    print(f'Calculating metrics for experiment {es.experiment_number}...')
    if es.experiment_number == 1:
        # In this case, threshold = number of labels, so we can consider that all responses
        # are for the correct labels
        total = (
            behaviour[constants.correct_response_idx]
            + behaviour[constants.incorrect_response_idx]
            + behaviour[constants.no_response_idx]
        )
        responses = total - behaviour[constants.no_response_idx]
        # If there are no responses, precision is undefined. Let's set it to 1.0.
        precision = (
            1.0
            if responses == 0
            else behaviour[constants.correct_response_idx] / float(responses)
        )
        recall = responses / float(total)
        accuracy = behaviour[constants.correct_response_idx] / float(total)
    elif es.experiment_number == 2:
        TP = (
            behaviour[constants.correct_response_idx]
            + behaviour[constants.incorrect_response_idx]
        )
        FP = (
            behaviour[constants.correct_mis_response_idx]
            + behaviour[constants.incorrect_mis_response_idx]
        )
        FN = behaviour[constants.no_response_idx]
        TN = behaviour[constants.no_mis_response_idx]
        print(f'TP:{TP}, FN:{FN}')
        print(f'FP:{FP}, TN:{TN}')
        precision = 1.0 if TP + FP == 0 else TP / float(TP + FP)
        recall = TP / float(TP + FN)
        accuracy = (TN + TP) / float(TP + FP + TN + FN)
    else:
        raise ValueError(f'Unknown experiment number: {es.experiment_number}')
    return precision, recall, accuracy


def recognize_by_memory(eam, tef_rounded, tel, msize, qd, classifier, threshold, es):
    data = []
    labels = []
    unknown = constants.network_labels
    confrix = np.zeros(
        (constants.memory_labels, constants.network_labels + 1), dtype='int'
    )
    behaviour = np.zeros(constants.n_behaviours, dtype=np.float64)
    for features, label in zip(tef_rounded, tel):
        memory, recognized, _ = eam.recall(features)
        if recognized:
            mem = qd.dequantize(memory, msize)
            data.append(mem)
            labels.append(label)
        else:
            confrix[label, unknown] += 1
    if len(labels) > 0:
        data = np.array(data)
        predictions = np.argmax(classifier.predict(data), axis=1)
        for correct, prediction in zip(labels, predictions):
            confrix[correct, prediction] += 1
    print(
        f'Calculating responses for experiment {es.experiment_number} and threshold {threshold}...'
    )
    behaviour[constants.no_response_idx] = np.sum(confrix[:threshold, unknown])
    behaviour[constants.no_mis_response_idx] = np.sum(confrix[threshold:, unknown])
    behaviour[constants.correct_response_idx] = np.sum(
        [confrix[i, i] for i in range(threshold)]
    )
    behaviour[constants.correct_mis_response_idx] = np.sum(
        [confrix[i, i] for i in range(threshold, constants.memory_labels)]
    )
    behaviour[constants.incorrect_response_idx] = (
        np.sum(confrix[:threshold, :unknown])
        - behaviour[constants.correct_response_idx]
    )
    behaviour[constants.incorrect_mis_response_idx] = (
        np.sum(confrix[threshold:, :unknown])
        - behaviour[constants.correct_mis_response_idx]
    )
    print('Confusion matrix:')
    constants.print_csv(confrix)
    print('Behaviour:')
    constants.print_csv(behaviour)
    return confrix, behaviour


def ams_size_results(
    midx,
    msize,
    domain,
    filling_features,
    testing_features,
    filling_labels,
    testing_labels,
    classifier,
    threshold,
    es,
):
    """Analyze the results of the AMS experiment for the given memory size."""

    behaviour = np.zeros(constants.n_behaviours, dtype=np.float64)

    # Create the memory.
    eam = AssociativeMemory(
        domain,
        msize,
        es,
    )
    # Round the values after filtering them.
    qd = qudeq.QuDeq(filling_features, percentiles=constants.use_percentiles)
    ff_rounded = qd.quantize(filling_features, msize)
    tf_rounded = qd.quantize(testing_features, msize)
    print(f'Features to register shape = {ff_rounded.shape}')
    print(f'Testing features shape = {tf_rounded.shape}')
    print('--------------------------------------------')
    print('Filling the memory...', end='', flush=True)
    for features in ff_rounded:
        eam.register(features)
    print('done.')

    # Recognize test data.
    confrix, behaviour = recognize_by_memory(
        eam,
        tf_rounded,
        testing_labels,
        msize,
        qd,
        classifier,
        threshold,
        es,
    )
    precision, recall, accuracy = calculate_metrics(behaviour, es)
    behaviour[constants.precision_idx] = precision
    behaviour[constants.accuracy_idx] = accuracy
    behaviour[constants.recall_idx] = recall
    return midx, eam.entropy, behaviour, confrix


def test_memory_sizes(domain, es):
    all_entropies = []
    all_confrixes = []
    all_behaviours = []

    print('Testing the memory')

    # Retrieve de classifier
    model_prefix = constants.model_name(es)

    for fold in range(constants.n_folds):
        gc.collect()
        print(f'Fold: {fold}')
        # Loads the classifier neural network.
        filename = constants.classifier_filename(model_prefix, fold)
        classifier = tf.keras.models.load_model(filename)

        # Loads the full set of features and labels.
        threshold = constants.memory_labels // es.experiment_number
        filling_features, filling_labels, testing_features, testing_labels = (
            load_features_and_labels(threshold, es, fold)
        )
        print('Filtered data:')
        print(f'\tFilling data shape: {filling_features.shape}')
        print(f'\tTesting data shape: {testing_features.shape}')
        print(f'\tTotal of labels = {len(np.unique(testing_labels))}')

        behaviours = np.zeros((len(constants.memory_sizes), constants.n_behaviours))
        measures = []
        confrixes = []
        entropies = []
        for midx, msize in enumerate(constants.memory_sizes):
            print(f'Memory size: {msize}')
            results = ams_size_results(
                midx,
                msize,
                domain,
                filling_features,
                testing_features,
                filling_labels,
                testing_labels,
                classifier,
                threshold,
                es,
            )
            measures.append(results)
        for midx, entropy, behaviour, confrix in measures:
            entropies.append(entropy)
            behaviours[midx, :] = behaviour
            confrixes.append(confrix)

        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        all_entropies.append(entropies)
        all_confrixes.append(confrixes)
        all_behaviours.append(behaviours)

    # Every "row" is training fold, and every column is a memory size.
    all_entropies = np.array(all_entropies)
    all_confrixes = np.array(all_confrixes)
    all_behaviours = np.array(all_behaviours)

    mean_entropy = np.mean(all_entropies, axis=0)
    mean_precision = np.mean(
        all_behaviours[:, :, constants.precision_idx] * 100, axis=0
    )
    stdev_precision = np.std(
        all_behaviours[:, :, constants.precision_idx] * 100, axis=0
    )
    mean_recall = np.mean(all_behaviours[:, :, constants.recall_idx] * 100, axis=0)
    stdev_recall = np.std(all_behaviours[:, :, constants.recall_idx] * 100, axis=0)
    mean_accuracy = np.mean(all_behaviours[:, :, constants.accuracy_idx] * 100, axis=0)
    stdev_accuracy = np.std(all_behaviours[:, :, constants.accuracy_idx] * 100, axis=0)

    best_memory_idx = optimum_indexes(mean_precision, mean_accuracy)
    best_memory_sizes = [constants.memory_sizes[i] for i in best_memory_idx]
    mean_behaviours = np.mean(all_behaviours, axis=0)
    stdv_behaviours = np.std(all_behaviours, axis=0)
    mean_confrixes = np.mean(all_confrixes, axis=0)
    stdv_confrixes = np.std(all_confrixes, axis=0)

    np.savetxt(
        constants.csv_filename('memory_entropy', es, sub_dir=constants.n_labels_path),
        all_entropies,
        delimiter=',',
    )
    np.savetxt(
        constants.csv_filename('mean_behaviours', es, sub_dir=constants.n_labels_path),
        mean_behaviours,
        delimiter=',',
    )
    np.savetxt(
        constants.csv_filename('stdv_behaviours', es, sub_dir=constants.n_labels_path),
        stdv_behaviours,
        delimiter=',',
    )
    np.save(
        constants.data_filename('mean_confrixes', es, sub_dir=constants.n_labels_path),
        mean_confrixes,
    )
    np.save(
        constants.data_filename('stdv_confrixes', es, sub_dir=constants.n_labels_path),
        stdv_confrixes,
    )
    plot_metrics_graph(
        mean_precision,
        mean_recall,
        mean_accuracy,
        mean_entropy,
        stdev_precision,
        stdev_recall,
        stdev_accuracy,
        constants.memory_sizes,
        '_msizes',
        es,
        sub_dir=constants.n_labels_path,
    )
    plot_responses_graph(
        mean_behaviours,
        stdv_behaviours,
        constants.memory_sizes,
        '_msizes',
        es,
        sub_dir=constants.n_labels_path,
    )
    print('Memory size evaluation completed!')
    return best_memory_sizes


def test_filling_percent(
    eam,
    msize,
    qd,
    filling_features,
    testing_features,
    testing_labels,
    percent,
    classifier,
    threshold,
    es,
):
    # Registrate filling data.
    for features in filling_features:
        eam.register(features)
    print(f'Filling of memories done at {percent}%')
    _, behaviour = recognize_by_memory(
        eam, testing_features, testing_labels, msize, qd, classifier, threshold, es
    )
    precision, recall, accuracy = calculate_metrics(behaviour, es)
    behaviour[constants.precision_idx] = precision
    behaviour[constants.accuracy_idx] = accuracy
    behaviour[constants.recall_idx] = recall
    return behaviour, eam.entropy


def test_filling_per_fold(mem_size, domain, es, fold):
    # Create the required associative memories.
    eam = AssociativeMemory(
        domain,
        mem_size,
        es,
    )
    model_prefix = constants.model_name(es)
    filename = constants.classifier_filename(model_prefix, fold)
    classifier = tf.keras.models.load_model(filename)

    threshold = constants.memory_labels // es.experiment_number
    filling_features, filling_labels, testing_features, testing_labels = (
        load_features_and_labels(threshold, es, fold)
    )
    print('Filtered data:')
    print(f'Filling data shape: {filling_features.shape}')
    print(f'Testing data shape: {testing_features.shape}')

    qd = qudeq.QuDeq(filling_features, percentiles=constants.use_percentiles)
    filling_features = qd.quantize(filling_features, mem_size)
    testing_features = qd.quantize(testing_features, mem_size)

    total = len(filling_labels)
    percents = np.array(constants.memory_fills)
    steps = np.round(total * percents / 100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []
    fold_accuracy = []

    start = 0
    for percent, end in zip(percents, steps):
        features = filling_features[start:end]
        print(f'Filling from {start} to {end}.')
        behaviour, entropy = test_filling_percent(
            eam,
            mem_size,
            qd,
            features,
            testing_features,
            testing_labels,
            percent,
            classifier,
            threshold,
            es,
        )
        # A list of tuples (position, label, features)
        # fold_recalls += recalls
        # An array with average entropy per step.
        fold_entropies.append(entropy)
        # Arrays with precision, and recall.
        fold_precision.append(behaviour[constants.precision_idx])
        fold_recall.append(behaviour[constants.recall_idx])
        fold_accuracy.append(behaviour[constants.accuracy_idx])
        start = end
    # Use this to plot current state of memories
    # as heatmaps.
    # plot_memories(ams, es, fold)
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.array(fold_precision)
    fold_recall = np.array(fold_recall)
    fold_accuracy = np.array(fold_accuracy)
    print(f'Filling test completed for fold {fold}')
    return fold, fold_entropies, fold_precision, fold_recall, fold_accuracy


def test_memory_fills(domain, mem_sizes, es):
    memory_fills = constants.memory_fills
    testing_folds = constants.n_folds
    best_filling_percents = []
    for mem_size in mem_sizes:
        # All entropies, precision, and recall, per size, fold, and fill.
        total_entropies = np.zeros((testing_folds, len(memory_fills)))
        total_precisions = np.zeros((testing_folds, len(memory_fills)))
        total_recalls = np.zeros((testing_folds, len(memory_fills)))
        total_accuracies = np.zeros((testing_folds, len(memory_fills)))
        list_results = []

        for fold in range(testing_folds):
            results = test_filling_per_fold(mem_size, domain, es, fold)
            list_results.append(results)
        for fold, entropies, precisions, recalls, accuracies in list_results:
            total_precisions[fold] = precisions
            total_recalls[fold] = recalls
            total_accuracies[fold] = accuracies
            total_entropies[fold] = entropies

        main_avrge_entropies = np.mean(total_entropies, axis=0)
        main_stdev_entropies = np.std(total_entropies, axis=0)
        main_avrge_precisions = np.mean(total_precisions, axis=0)
        main_stdev_precisions = np.std(total_precisions, axis=0)
        main_avrge_recalls = np.mean(total_recalls, axis=0)
        main_stdev_recalls = np.std(total_recalls, axis=0)
        main_avrge_accuracies = np.mean(total_accuracies, axis=0)
        main_stdev_accuracies = np.std(total_accuracies, axis=0)

        suffix = '_mfills' + constants.numeric_suffix('sze', mem_size)
        np.savetxt(
            constants.csv_filename(
                'main_average_precision' + suffix,
                es,
                sub_dir=constants.n_labels_path,
            ),
            main_avrge_precisions,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_average_recall' + suffix,
                es,
                sub_dir=constants.n_labels_path,
            ),
            main_avrge_recalls,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_average_accuracy' + suffix,
                es,
                sub_dir=constants.n_labels_path,
            ),
            main_avrge_accuracies,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_average_entropy' + suffix,
                es,
                sub_dir=constants.n_labels_path,
            ),
            main_avrge_entropies,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_stdev_precision' + suffix,
                es,
                sub_dir=constants.n_labels_path,
            ),
            main_stdev_precisions,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_stdev_recall' + suffix,
                es,
                sub_dir=constants.n_labels_path,
            ),
            main_stdev_recalls,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_stdev_accuracy' + suffix,
                es,
                sub_dir=constants.n_labels_path,
            ),
            main_stdev_accuracies,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_stdev_entropy' + suffix,
                es,
                sub_dir=constants.n_labels_path,
            ),
            main_stdev_entropies,
            delimiter=',',
        )

        plot_metrics_graph(
            main_avrge_precisions * 100,
            main_avrge_recalls * 100,
            main_avrge_accuracies * 100,
            main_avrge_entropies,
            main_stdev_precisions * 100,
            main_stdev_recalls * 100,
            main_stdev_accuracies * 100,
            constants.memory_fills,
            '_mfills' + suffix,
            es,
            xtitle=_('Percentage of memory corpus'),
            sub_dir=constants.n_labels_path,
        )

        bf_idx = optimum_indexes(main_avrge_precisions, main_avrge_accuracies)
        best_filling_percents.append(constants.memory_fills[bf_idx[0]])
        print(f'Testing fillings for memory size {mem_size} done.')
    return best_filling_percents


# endregion Memory evaluation functions ---------------------------------------------------

# region Remembering functions ----------------------------------------------------------


def store_image(filename, array):
    pixels = array.reshape(dataset.rows, dataset.columns)
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(filename)


def store_original_and_test(testing, prod_test, directory, idx, label, es, fold):
    testing_filename = constants.testing_image_filename(directory, idx, label, es, fold)
    prod_test_filename = constants.prod_testing_image_filename(
        directory, idx, label, es, fold
    )
    for file in [
        testing_filename,
        prod_test_filename,
    ]:
        dirname = os.path.dirname(file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    store_image(testing_filename, testing)
    store_image(prod_test_filename, prod_test)


def store_memory(memory, directory, idx, label, es, fold):
    filename = constants.memory_image_filename(directory, idx, label, es, fold)
    full_directory = constants.dirname(filename)
    if full_directory not in store_memory.created_dirs:
        constants.create_directory(full_directory)
        store_memory.created_dirs.append(full_directory)
    store_image(filename, memory)


store_memory.created_dirs = []


def decode_test_features(es):
    """Creates images directly from test features, completing an autoencoder.

    Uses the decoder part of the neural networks to (re)create images from features
    generated by the encoder.
    """
    model_prefix = constants.model_name(es)
    testing_features_prefix = constants.features_prefix + constants.testing_suffix

    for fold in range(constants.n_folds):
        # Load test features and labels
        testing_features_filename = constants.data_filename(
            testing_features_prefix, es, fold
        )
        testing_features = np.load(testing_features_filename)
        testing_data, testing_labels = dataset.get_testing(fold)

        # Loads the decoder.
        model_filename = constants.decoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(model_filename)
        model.summary()

        prod_test_images = model.predict(testing_features) * 255
        testing_data *= 255
        n = len(testing_labels)

        for i, testing, prod_test, label in zip(
            range(n),
            testing_data,
            prod_test_images,
            testing_labels,
        ):
            store_original_and_test(
                testing,
                prod_test,
                constants.testing_path,
                i,
                label,
                es,
                fold,
            )


def decode_memories(msize, es):
    msize_suffix = constants.msize_suffix(msize)
    model_prefix = constants.model_name(es)
    testing_labels_prefix = constants.labels_prefix + constants.testing_suffix
    print(f'Running remembering for sigma = {es.mem_params[constants.sigma_idx]:.2f}')
    suffix = msize_suffix
    memories_prefix = constants.memories_name(es) + suffix
    for fold in range(constants.n_folds):
        # Load test features and labels
        memories_features_filename = constants.data_filename(memories_prefix, es, fold)
        testing_labels_filename = constants.data_filename(
            testing_labels_prefix, es, fold
        )
        memories_features = np.load(memories_features_filename)
        testing_labels = np.load(testing_labels_filename)
        # Loads the decoder.
        model_filename = constants.decoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(model_filename)
        model.summary()

        memories_images = model.predict(memories_features)
        n = len(testing_labels)
        memories_path = constants.memories_path + suffix
        for i, memory, label in zip(range(n), memories_images, testing_labels):
            store_memory(memory, memories_path, i, label, es, fold)


def remember_with_sigma(eam, features, prefixes, msize, qd, es, fold):
    memories_prefix = prefixes[0]
    recognition_prefix = prefixes[1]
    weights_prefix = prefixes[2]
    classif_prefix = prefixes[3]

    memories_features = []
    memories_recognition = []
    memories_weights = []
    for fs in features:
        memory, recognized, weight = eam.recall(fs)
        memories_features.append(memory)
        memories_recognition.append(recognized)
        memories_weights.append(weight)
    memories_features = np.array(memories_features, dtype=float)
    memories_features = qd.dequantize(memories_features, msize)
    memories_recognition = np.array(memories_recognition, dtype=int)
    memories_weights = np.array(memories_weights, dtype=float)

    model_prefix = constants.model_name(es)
    filename = constants.classifier_filename(model_prefix, fold)
    classifier = tf.keras.models.load_model(filename)
    classification = np.argmax(classifier.predict(memories_features), axis=1)
    for i in range(len(classification)):
        # If the memory does not recognize it, it should not be classified.
        if not memories_recognition[i]:
            classification[i] = constants.memory_labels

    features_filename = constants.data_filename(memories_prefix, es, fold)
    recognition_filename = constants.data_filename(recognition_prefix, es, fold)
    weights_filename = constants.data_filename(weights_prefix, es, fold)
    classification_filename = constants.data_filename(classif_prefix, es, fold)
    np.save(features_filename, memories_features)
    np.save(recognition_filename, memories_recognition)
    np.save(weights_filename, memories_weights)
    np.save(classification_filename, classification)


def remember(msize, mfill, es):
    msize_suffix = constants.msize_suffix(msize)
    print(f'Running remembering for sigma = {es.mem_params[constants.sigma_idx]}')
    suffix = msize_suffix
    memories_prefix = constants.memories_name(es) + suffix
    recognition_prefix = constants.recognition_name(es) + suffix
    weights_prefix = constants.weights_name(es) + suffix
    classif_prefix = constants.classification_name(es) + suffix
    prefixes_list = [
        [memories_prefix, recognition_prefix, weights_prefix, classif_prefix],
    ]

    for fold in range(constants.n_folds):
        print(f'Running remembering for fold: {fold}')
        # Load filling and testing features and labels
        filling_features, _, testing_features, _ = load_features_and_labels(es, fold)

        qd = qudeq.QuDeq(filling_features, percentiles=constants.use_percentiles)
        filling_rounded = qd.quantize(filling_features, msize)
        testing_rounded = qd.quantize(testing_features, msize)

        # Create the memory and fill it
        eam = AssociativeMemory(
            constants.domain,
            msize,
            es,
        )
        end = round(len(filling_features) * mfill / 100.0)
        for features in filling_rounded[:end]:
            eam.register(features)
        print(f'Memory of size {msize} filled with {end} elements for fold {fold}')

        for features, prefixes in zip([testing_rounded], prefixes_list):
            remember_with_sigma(eam, features, prefixes, msize, qd, es, fold)
    print('Remembering done!')


# endregion Remembering functions --------------------------------------------------------

# region Main functionality functions ----------------------------------------------------


def create_and_train_network(_):
    model_prefix = constants.model_name()
    stats_prefix = model_prefix + constants.classifier_suffix
    history, conf_matrix = neural_net.train_network(model_prefix)
    save_history(history, stats_prefix)
    save_conf_matrix(conf_matrix, stats_prefix)


def produce_features_from_data(_):
    model_prefix = constants.model_name()
    features_prefix = constants.features_name()
    labels_prefix = constants.labels_name()
    neural_net.obtain_features(model_prefix, features_prefix, labels_prefix)


def run_evaluation(es):
    """Run evaluation for the given experimental settings.

    The experiments test different memory sizes, and them choose the best
    ones. Then they evaluate filling percentages and them choose the best
    ones too.
    """
    best_memory_sizes = test_memory_sizes(constants.domain, es)
    print(f'Best memory sizes: {best_memory_sizes}')
    best_filling_percents = test_memory_fills(constants.domain, best_memory_sizes, es)
    save_learned_params(best_memory_sizes, best_filling_percents, es)


def generate_memories(es):
    decode_test_features(es)
    learned = load_learned_params(es)
    for msize, mfill in learned:
        remember(msize, mfill, es)
        decode_memories(msize, es)


# endregion Main functionality functions -------------------------------------------------

# region Command-line interface -----------------------------------------------------------

if __name__ == '__main__':
    args = docopt(__doc__)

    # Processing language.
    lang = 'en'
    if args['es']:
        lang = 'es'
        es = gettext.translation('eam', localedir='locale', languages=['es'])
        es.install()

    # Processing memory size (columns)
    if args['--domain']:
        constants.domain = int(args['--domain'])
        assert 0 < constants.domain

    # Processing runpath.
    if args['--runpath']:
        constants.run_path = args['--runpath']

    # Processing number of classes.
    num_classes = constants.memory_labels
    if args['--num-classes']:
        try:
            num_classes = int(args['--num-classes'])
            constants.set_memory_labels(num_classes)
            constants.n_labels_path = (
                constants.run_prefix + '_' + constants.int_suffix(num_classes)
            )
        except ValueError as e:
            print(e)
            sys.exit(1)

    prefix = constants.memory_parameters_prefix
    filename = constants.csv_filename(prefix)
    parameters = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
    exp_settings = constants.ExperimentSettings(params=parameters)
    print(f'Working directory: {constants.run_path}')
    print(f'Subworking directory: {constants.n_labels_path}')
    print(f'Experimental settings: {exp_settings}')
    print(f'Memory size (columns): {constants.domain}')
    print(f'Number of classes: {num_classes}')

    # PROCESSING OF MAIN OPTIONS.

    if args['-n']:
        create_and_train_network(exp_settings)
    elif args['-f']:
        produce_features_from_data(exp_settings)
    elif args['-e']:
        experiment = int(args['<experiment>'])
        if (experiment < 1) or (experiment > 2):
            print(f'Experiment number not valid: {experiment}')
            sys.exit(1)

        if (experiment == 2) and ((num_classes % 2) != 0):
            print(f'Invalid classes number: {num_classes}, must be an even number')
            sys.exit(1)

        exp_settings.experiment_number = experiment
        print(f'Running experiment {experiment} with {num_classes} classes.')
        run_evaluation(exp_settings)
    elif args['-r']:
        generate_memories(exp_settings)


# endregion Command-line interface ---------------------------------------------------------
