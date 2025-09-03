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
  eam (-n | -f | -e <experiment> | -r) [--num-classes=NUM] [--domain=DOMAIN] [--runpath=PATH ] [ -l (en | es) ]

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
from itertools import islice
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


def plot_pre_graph(
    pre_mean,
    acc_mean,
    ent_mean,
    pre_std,
    acc_std,
    es,
    tag='',
    xlabels=constants.memory_sizes,
    xtitle=None,
    ytitle=None,
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

    plt.errorbar(x, pre_mean, fmt='r-o', yerr=pre_std, label=_('Precision'))
    plt.errorbar(x, acc_mean, fmt='b--s', yerr=acc_std, label=_('Accuracy'))
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

    s = tag + '-graph_prse_MEAN' + _('-english')
    graph_filename = constants.picture_filename(s, es, None)
    plt.savefig(graph_filename, dpi=600)


def plot_size_graph(response_size, size_stdev, es):
    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length / len(response_size)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = constants.n_labels

    plt.errorbar(
        x,
        response_size,
        fmt='g-D',
        yerr=size_stdev,
        label=_('Average number of responses'),
    )
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, constants.memory_sizes)
    plt.yticks(np.arange(0, ymax + 1, 1), range(constants.n_labels + 1))

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Size'))
    plt.legend(loc=1)
    plt.grid(True)

    graph_filename = constants.picture_filename('graph_size_MEAN' + _('-english'), es)
    plt.savefig(graph_filename, dpi=600)


def plot_behs_graph(
    incorrect_no_response,
    incorrect,
    correct,
    es,
    correct_no_response=None,
    xtitle=None,
    ytitle=None,
):
    if correct_no_response is None:
        correct_no_response = np.zeros_like(incorrect_no_response)
    for i in range(len(incorrect_no_response)):
        total = (
            incorrect_no_response[i]
            + correct_no_response[i]
            + incorrect[i]
            + correct[i]
        ) / 100.0
        correct_no_response[i] /= total
        incorrect_no_response
        incorrect[i] /= total
        correct[i] /= total

    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length / len(constants.memory_sizes)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5  # the width of the bars: can also be len(x) sequence

    plt.bar(x, correct, width, label=_('Correct response'))
    cumm = np.array(correct)
    plt.bar(x, incorrect, width, bottom=cumm, label=_('Incorrect response'))
    cumm += np.array(incorrect)
    plt.bar(x, correct_no_response, width, bottom=cumm, label=_('Correct no response'))
    cumm += np.array(correct_no_response)
    plt.bar(
        x, incorrect_no_response, width, bottom=cumm, label=_('Incorrect no response')
    )
    cumm += np.array(incorrect_no_response)

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, constants.memory_sizes)

    if xtitle is None:
        xtitle = _('Range Quantization Levels')
    if ytitle is None:
        ytitle = _('Labels')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.legend(loc=0)
    plt.grid(axis='y')

    graph_filename = constants.picture_filename(
        'graph_behaviours_MEAN' + _('-english'), es, None
    )
    plt.savefig(graph_filename, dpi=600)


def plot_features_graph(domain, means, stdevs, es):
    """Draws the characterist shape of features per label.

    The graph is a dots and lines graph with error bars denoting standard deviations.
    """
    ymin = np.PINF
    ymax = np.NINF
    for i in constants.all_n_labels:
        yn = (means[i] - stdevs[i]).min()
        yx = (means[i] + stdevs[i]).max()
        ymin = ymin if ymin < yn else yn
        ymax = ymax if ymax > yx else yx
    main_step = 100.0 / domain
    xrange = np.arange(0, 100, main_step)
    fmts = constants.label_formats
    for i in constants.all_n_labels:
        plt.clf()
        plt.figure(figsize=(12, 5))
        plt.errorbar(xrange, means[i], fmt=fmts[i], yerr=stdevs[i], label=str(i))
        plt.xlim(0, 100)
        plt.ylim(ymin, ymax)
        plt.xticks(xrange, labels='')
        plt.xlabel(_('Features'))
        plt.ylabel(_('Values'))
        plt.legend(loc='right')
        plt.grid(True)
        filename = constants.features_name(es) + '-' + str(i).zfill(3) + _('-english')
        plt.savefig(constants.picture_filename(filename, es), dpi=600)


def plot_conf_matrix(matrix, xtags, ytags, prefix, es):
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
    filename = constants.picture_filename(prefix, es)
    plt.savefig(filename, dpi=600)


def plot_memory(memory: AssociativeMemory, prefix, es, fold):
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
    filename = constants.picture_filename(prefix, es, fold)
    plt.savefig(filename, dpi=600)


def maximum(arrays):
    max = float('-inf')
    for a in arrays:
        local_max = np.max(a)
        if local_max > max:
            max = local_max
    return max


def minimum(arrays):
    min = float('inf')
    for a in arrays:
        local_min = np.min(a)
        if local_min < min:
            min = local_min
    return min


def recognize_by_memory(eam, tef_rounded, tel, msize, qd, classifier, threshold, es):
    data = []
    labels = []
    unknown = constants.n_labels
    confrix = np.zeros((constants.n_labels, constants.n_labels + 1), dtype='int')
    behaviour = np.zeros(constants.n_behaviours, dtype=np.float64)
    for features, label in zip(tef_rounded, tel):
        memory, recognized, _ = eam.recall(features)
        if recognized:
            mem = qd.dequantize(memory, msize)
            data.append(mem)
            labels.append(label)
        else:
            confrix[label, unknown] += 1
    data = np.array(data)
    predictions = np.argmax(classifier.predict(data), axis=1)
    for correct, prediction in zip(labels, predictions):
        confrix[correct, prediction] += 1
    if es.experiment_number == 1:
        behaviour[constants.no_response_idx] = np.sum(confrix[:, unknown])
    elif es.experiment_number == 2:
        behaviour[constants.correct_no_response_idx] = np.sum(
            confrix[threshold:, unknown]
        )
        behaviour[constants.incorrect_no_response_idx] = np.sum(
            confrix[:threshold, unknown]
        )
    behaviour[constants.correct_response_idx] = np.sum(
        [confrix[i, i] for i in range(threshold)]
    )

    behaviour[constants.incorrect_response_idx] = (
        len(tel)
        - np.sum(confrix[:, unknown])
        - behaviour[constants.correct_response_idx]
    )
    print('Confusion matrix:')
    constants.print_csv(confrix)
    return confrix, behaviour


def split_by_label(fl_pairs):
    label_dict = {}
    for label in range(constants.n_labels):
        label_dict[label] = []
    for features, label in fl_pairs:
        label_dict[label].append(features)
    return label_dict.items()


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def optimum_indexes(precisions, accuracies):
    accs = []
    for idx, acc in enumerate(accuracies):
        accs.append((acc, idx))
    accs.sort(reverse=True, key=lambda tuple: tuple[0])
    return [t[1] for t in accs[: constants.n_best_memory_sizes]]


def ams_size_results(
    midx,
    msize,
    domain,
    filling_features,
    testing_features,
    filling_labels,
    testing_labels,
    classifier,
    es,
):
    """Analyze the results of the AMS experiment for the given memory size."""
    for i in range(5):
        print(' ')

    np.set_printoptions(threshold=200)

    print('--------------------------------------------')
    print(f'n_labels = {constants.n_labels}')
    print('--------------------------------------------')

    # Round the values
    qd = qudeq.QuDeq(filling_features, percentiles=constants.use_percentiles)
    ff_rounded = qd.quantize(filling_features, msize)
    tf_rounded = qd.quantize(testing_features, msize)

    behaviour = np.zeros(constants.n_behaviours, dtype=np.float64)

    # Create the memory.
    eam = AssociativeMemory(
        domain,
        msize,
        es,
    )

    known_threshold = constants.n_labels
    if es.experiment_number == 2:
        known_threshold //= 2
        print('--------------------------------------------')
        print(f'Adjusted known_threshold = {known_threshold}')
        print('--------------------------------------------')

        known_labels_mask = filling_labels < known_threshold
        ff_rounded = ff_rounded[known_labels_mask]
    print('--------------------------------------------')
    print(f'Filling features shape = {filling_features.shape}')
    print(f'Features to register shape = {ff_rounded.shape}')
    print(f'Testing features shape = {tf_rounded.shape}')
    print('--------------------------------------------')

    for features in ff_rounded:
        eam.register(features)

    # Recognize test data.
    confrix, behaviour = recognize_by_memory(
        eam,
        tf_rounded,
        testing_labels,
        msize,
        qd,
        classifier,
        known_threshold,
        es,
    )

    # If there are no responses, precision is undefined. Let's set it to 0.
    if es.experiment_number == 1:
        responses = len(testing_labels) - behaviour[constants.no_response_idx]
        if responses > 0:
            precision = behaviour[constants.correct_response_idx] / float(responses)
        else:
            precision = 0.0  # Avoid division by zero

        accuracy = behaviour[constants.correct_response_idx] / float(
            len(testing_labels)
        )
    elif es.experiment_number == 2:
        true_positives = behaviour[constants.correct_response_idx]
        false_positives = behaviour[constants.incorrect_response_idx]
        true_negatives = behaviour[constants.correct_no_response_idx]
        if true_positives + false_positives > 0:
            precision = true_positives / float(true_positives + false_positives)
        else:
            precision = 0.0  # Avoid division by zero
        accuracy = (true_negatives + true_positives) / float(len(testing_labels))

    behaviour[constants.precision_idx] = precision
    behaviour[constants.accuracy_idx] = accuracy
    return midx, eam.entropy, behaviour, confrix


def test_memory_sizes(domain, es):
    all_entropies = []
    precision = []
    accuracy = []
    all_confrixes = []

    no_response = []
    correct_no_response = []
    incorrect_no_response = []
    incorrect_response = []
    correct_response = []

    print('Testing the memory')

    # Retrieve de classifier
    model_prefix = constants.model_name(es)

    for fold in range(constants.n_folds):
        gc.collect()
        filename = constants.classifier_filename(model_prefix, es, fold)
        classifier = tf.keras.models.load_model(filename)
        print(f'Fold: {fold}')
        suffix = constants.filling_suffix
        filling_features_filename = constants.features_name(es) + suffix
        filling_features_filename = constants.input_data_filename(
            filling_features_filename, es, fold
        )
        filling_labels_filename = constants.labels_name(es) + suffix
        filling_labels_filename = constants.input_data_filename(
            filling_labels_filename, es, fold
        )

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name(es) + suffix
        testing_features_filename = constants.input_data_filename(
            testing_features_filename, es, fold
        )
        testing_labels_filename = constants.labels_name(es) + suffix
        testing_labels_filename = constants.input_data_filename(
            testing_labels_filename, es, fold
        )

        filling_features = np.load(filling_features_filename)
        filling_labels = np.load(filling_labels_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)

        print(f'Filling data shape: {filling_features.shape}')
        print(f'Testing data shape: {testing_features.shape}')

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

        # Average precision and recall as percentage
        precision.append(behaviours[:, constants.precision_idx] * 100)
        accuracy.append(behaviours[:, constants.accuracy_idx] * 100)

        all_confrixes.append(np.array(confrixes))
        no_response.append(behaviours[:, constants.no_response_idx])
        correct_no_response.append(behaviours[:, constants.correct_no_response_idx])
        incorrect_no_response.append(behaviours[:, constants.incorrect_no_response_idx])
        incorrect_response.append(behaviours[:, constants.incorrect_response_idx])
        correct_response.append(behaviours[:, constants.correct_response_idx])

    # Every row is training fold, and every column is a memory size.
    all_entropies = np.array(all_entropies)
    precision = np.array(precision)
    accuracy = np.array(accuracy)
    all_confrixes = np.array(all_confrixes)

    average_entropy = np.mean(all_entropies, axis=0)
    average_precision = np.mean(precision, axis=0)
    stdev_precision = np.std(precision, axis=0)
    average_accuracy = np.mean(accuracy, axis=0)
    stdev_accuracy = np.std(accuracy, axis=0)
    average_confrixes = np.mean(all_confrixes, axis=0)

    no_response = np.array(no_response)
    correct_no_response = np.array(correct_no_response)
    incorrect_no_response = np.array(incorrect_no_response)
    incorrect_response = np.array(incorrect_response)
    correct_response = np.array(correct_response)
    mean_no_response = np.mean(no_response, axis=0)
    stdv_no_response = np.std(no_response, axis=0)
    mean_correct_no_response = np.mean(correct_no_response, axis=0)
    stdv_correct_no_response = np.std(correct_no_response, axis=0)
    mean_incorrect_no_response = np.mean(incorrect_no_response, axis=0)
    stdv_incorrect_no_response = np.std(incorrect_no_response, axis=0)
    mean_incorrect_response = np.mean(incorrect_response, axis=0)
    stdv_incorrect_response = np.std(incorrect_response, axis=0)
    mean_correct_response = np.mean(correct_response, axis=0)
    stdv_correct_response = np.std(correct_response, axis=0)
    best_memory_idx = optimum_indexes(average_precision, average_accuracy)
    best_memory_sizes = [constants.memory_sizes[i] for i in best_memory_idx]
    mean_behaviours = [
        mean_no_response,
        mean_correct_no_response,
        mean_incorrect_no_response,
        mean_incorrect_response,
        mean_correct_response,
    ]
    stdv_behaviours = [
        stdv_no_response,
        stdv_correct_no_response,
        stdv_incorrect_no_response,
        stdv_incorrect_response,
        stdv_correct_response,
    ]

    np.savetxt(
        constants.csv_filename('memory_precision', es, None),
        precision,
        delimiter=',',
    )
    np.savetxt(
        constants.csv_filename('memory_recall', es, None),
        accuracy,
        delimiter=',',
    )
    np.savetxt(
        constants.csv_filename('memory_entropy', es, None),
        all_entropies,
        delimiter=',',
    )
    np.savetxt(
        constants.csv_filename('mean_behaviours', es, None),
        mean_behaviours,
        delimiter=',',
    )
    np.savetxt(
        constants.csv_filename('stdv_behaviours', es, None),
        stdv_behaviours,
        delimiter=',',
    )
    np.save(
        constants.data_filename('memory_confrixes', es, None),
        average_confrixes,
    )
    np.save(
        constants.data_filename('behaviours', es, None),
        behaviours,
    )
    plot_pre_graph(
        average_precision,
        average_accuracy,
        average_entropy,
        stdev_precision,
        stdev_accuracy,
        es,
    )
    plot_behs_graph(
        mean_incorrect_no_response,
        mean_incorrect_response,
        mean_correct_response,
        es,
        correct_no_response=mean_correct_no_response,
        xtitle='Memory size (rows)',
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
    # If there are no responses, precision is undefined. Let's set it to 0.
    if es.experiment_number == 1:
        responses = len(testing_labels) - behaviour[constants.no_response_idx]
        if responses > 0:
            precision = behaviour[constants.correct_response_idx] / float(responses)
        else:
            precision = 0.0  # Avoid division by zero

        accuracy = behaviour[constants.correct_response_idx] / float(
            len(testing_labels)
        )
    elif es.experiment_number == 2:
        true_positives = behaviour[constants.correct_response_idx]
        false_positives = behaviour[constants.incorrect_response_idx]
        true_negatives = behaviour[constants.correct_no_response_idx]
        if true_positives + false_positives > 0:
            precision = true_positives / float(true_positives + false_positives)
        else:
            precision = 0.0  # Avoid division by zero
        accuracy = (true_negatives + true_positives) / float(len(testing_labels))

    behaviour[constants.precision_idx] = precision
    behaviour[constants.accuracy_idx] = accuracy
    return behaviour, eam.entropy


def test_filling_per_fold(mem_size, domain, es, fold):
    # Create the required associative memories.
    eam = AssociativeMemory(
        domain,
        mem_size,
        es,
    )
    model_prefix = constants.model_name(es)
    filename = constants.classifier_filename(model_prefix, es, fold)
    classifier = tf.keras.models.load_model(filename)

    suffix = constants.filling_suffix
    filling_features_filename = constants.features_name(es) + suffix
    filling_features_filename = constants.input_data_filename(
        filling_features_filename, es, fold
    )
    filling_labels_filename = constants.labels_name(es) + suffix
    filling_labels_filename = constants.input_data_filename(
        filling_labels_filename, es, fold
    )

    suffix = constants.testing_suffix
    testing_features_filename = constants.features_name(es) + suffix
    testing_features_filename = constants.input_data_filename(
        testing_features_filename, es, fold
    )
    testing_labels_filename = constants.labels_name(es) + suffix
    testing_labels_filename = constants.input_data_filename(
        testing_labels_filename, es, fold
    )

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    # Filter the data to include only the classes for the current experiment
    filling_mask = filling_labels < constants.n_labels
    filling_features = filling_features[filling_mask]
    filling_labels = filling_labels[filling_mask]

    qd = qudeq.QuDeq(filling_features, percentiles=constants.use_percentiles)
    known_threshold = constants.n_labels
    print('--------------------------------------------')
    print(f'known_threshold = {known_threshold}')
    print('--------------------------------------------')

    if es.experiment_number == 2:
        known_threshold //= 2
        print('--------------------------------------------')
        print(f'Adjusted known_threshold = {known_threshold}')
        print('--------------------------------------------')

        known_label_mask = filling_labels < known_threshold
        filling_features = filling_features[known_label_mask]
    print('--------------------------------------------')
    print(f'filling labels shape = {filling_labels.shape}')
    print(f'filling features shape = {filling_features.shape}')
    print('--------------------------------------------')
    filling_features = qd.quantize(filling_features, mem_size)
    testing_features = qd.quantize(testing_features, mem_size)

    total = len(filling_labels)
    percents = np.array(constants.memory_fills)
    steps = np.round(total * percents / 100.0).astype(int)

    fold_entropies = []
    fold_precision = []
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
            known_threshold,
            es,
        )
        # A list of tuples (position, label, features)
        # fold_recalls += recalls
        # An array with average entropy per step.
        fold_entropies.append(entropy)
        # Arrays with precision, and recall.
        fold_precision.append(behaviour[constants.precision_idx])
        fold_accuracy.append(behaviour[constants.accuracy_idx])
        start = end
    # Use this to plot current state of memories
    # as heatmaps.
    # plot_memories(ams, es, fold)
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.array(fold_precision)
    fold_accuracy = np.array(fold_accuracy)
    print(f'Filling test completed for fold {fold}')
    return fold, fold_entropies, fold_precision, fold_accuracy


def test_memory_fills(domain, mem_sizes, es):
    memory_fills = constants.memory_fills
    testing_folds = constants.n_folds
    best_filling_percents = []
    for mem_size in mem_sizes:
        # All entropies, precision, and recall, per size, fold, and fill.
        total_entropies = np.zeros((testing_folds, len(memory_fills)))
        total_precisions = np.zeros((testing_folds, len(memory_fills)))
        total_accuracies = np.zeros((testing_folds, len(memory_fills)))
        list_results = []

        for fold in range(testing_folds):
            results = test_filling_per_fold(mem_size, domain, es, fold)
            list_results.append(results)
        for fold, entropies, precisions, accuracies in list_results:
            total_precisions[fold] = precisions
            total_accuracies[fold] = accuracies
            total_entropies[fold] = entropies

        main_avrge_entropies = np.mean(total_entropies, axis=0)
        main_stdev_entropies = np.std(total_entropies, axis=0)
        main_avrge_precisions = np.mean(total_precisions, axis=0)
        main_stdev_precisions = np.std(total_precisions, axis=0)
        main_avrge_accuracies = np.mean(total_accuracies, axis=0)
        main_stdev_accuracies = np.std(total_accuracies, axis=0)

        suffix = constants.numeric_suffix(
            'exp', es.experiment_number
        ) + constants.numeric_suffix('sze', mem_size)
        np.savetxt(
            constants.csv_filename(
                'main_average_precision' + suffix,
                es,
                None,
            ),
            main_avrge_precisions,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_average_recall' + suffix,
                es,
                None,
            ),
            main_avrge_accuracies,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_average_entropy' + suffix,
                es,
                None,
            ),
            main_avrge_entropies,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_stdev_precision' + suffix,
                es,
                None,
            ),
            main_stdev_precisions,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_stdev_recall' + suffix,
                es,
                None,
            ),
            main_stdev_accuracies,
            delimiter=',',
        )
        np.savetxt(
            constants.csv_filename(
                'main_stdev_entropy' + suffix,
                es,
                None,
            ),
            main_stdev_entropies,
            delimiter=',',
        )

        plot_pre_graph(
            main_avrge_precisions * 100,
            main_avrge_accuracies * 100,
            main_avrge_entropies,
            main_stdev_precisions * 100,
            main_stdev_accuracies * 100,
            es,
            'recall' + suffix,
            xlabels=constants.memory_fills,
            xtitle=_('Percentage of memory corpus'),
        )

        bf_idx = optimum_indexes(main_avrge_precisions, main_avrge_accuracies)
        best_filling_percents.append(constants.memory_fills[bf_idx[0]])
        print(f'Testing fillings for memory size {mem_size} done.')
    return best_filling_percents


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


def save_conf_matrix(matrix, prefix, es):
    name = prefix + constants.matrix_suffix
    plot_conf_matrix(
        matrix, range(constants.n_labels), range(constants.n_labels), name, es
    )
    filename = constants.data_filename(name, es)
    np.save(filename, matrix)


def save_learned_params(mem_sizes, fill_percents, es):
    name = constants.learn_params_name(es)
    filename = constants.data_filename(name, es, None, )
    np.save(filename, np.array([mem_sizes, fill_percents], dtype=int))


def load_learned_params(es):
    name = constants.learn_params_name(es)
    filename = constants.data_filename(name, es)
    params = np.load(filename)
    size_fill = [(params[0, j], params[1, j]) for j in range(params.shape[1])]
    return size_fill


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
        suffix = constants.filling_suffix
        filling_features_filename = constants.features_name(es) + suffix
        filling_features_filename = constants.input_data_filename(
            filling_features_filename, es, fold
        )

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name(es) + suffix
        testing_features_filename = constants.input_data_filename(
            testing_features_filename, es, fold
        )

        filling_features = np.load(filling_features_filename)
        testing_features = np.load(testing_features_filename)
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
    filename = constants.classifier_filename(model_prefix, es, fold)
    classifier = tf.keras.models.load_model(filename)
    classification = np.argmax(classifier.predict(memories_features), axis=1)
    for i in range(len(classification)):
        # If the memory does not recognize it, it should not be classified.
        if not memories_recognition[i]:
            classification[i] = constants.n_labels

    features_filename = constants.data_filename(memories_prefix, es, fold)
    recognition_filename = constants.data_filename(recognition_prefix, es, fold)
    weights_filename = constants.data_filename(weights_prefix, es, fold)
    classification_filename = constants.data_filename(classif_prefix, es, fold)
    np.save(features_filename, memories_features)
    np.save(recognition_filename, memories_recognition)
    np.save(weights_filename, memories_weights)
    np.save(classification_filename, classification)


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
    constants.create_directory(full_directory)
    store_image(filename, memory)


def store_dream(dream, label, index, suffix, es, fold):
    dreams_path = constants.dreams_path + suffix
    store_memory(dream, dreams_path, index, label, es, fold)


def store_image(filename, array):
    pixels = array.reshape(dataset.columns, dataset.rows)
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(filename)


##############################################################################
# Main section


def create_and_train_network(es):
    model_prefix = constants.model_name(es)
    stats_prefix = model_prefix + constants.classifier_suffix
    history, conf_matrix = neural_net.train_network(model_prefix, es)
    save_history(history, stats_prefix, es)
    save_conf_matrix(conf_matrix, stats_prefix, es)


def produce_features_from_data(es):
    model_prefix = constants.model_name(es)
    features_prefix = constants.features_name(es)
    labels_prefix = constants.labels_name(es)
    data_prefix = constants.data_name(es)
    neural_net.obtain_features(
        model_prefix, features_prefix, labels_prefix, data_prefix, es
    )


def run_evaluation(es):
    """Run evaluation for the given experimental settings.

    The first experiment tests different memory sizes,
    and it chooses the best ones. Then it evaluates filling percentages and
    it chooses the best ones too.

    The second experiment assumes the best memory sizes have been selected,
    and it only evaluates filling percentages, but it does not stores them.
    """
    if es.experiment_number == 1:
        best_memory_sizes = test_memory_sizes(constants.domain, es)
        print(f'Best memory sizes: {best_memory_sizes}')
        best_filling_percents = test_memory_fills(
            constants.domain, best_memory_sizes, es
        )
        save_learned_params(best_memory_sizes, best_filling_percents, es)
    elif es.experiment_number == 2:
        # Load the learned parameters.
        learned = load_learned_params(es)
        best_memory_sizes = [msize for msize, _ in learned]
        print(f'Best memory sizes: {best_memory_sizes}')
        # Evaluate the memories with the learned parameters.
        test_memory_fills(constants.domain, best_memory_sizes, es)


def generate_memories(es):
    decode_test_features(es)
    learned = load_learned_params(es)
    for msize, mfill in learned:
        remember(msize, mfill, es)
        decode_memories(msize, es)


if __name__ == '__main__':
    args = docopt(__doc__)

    # Processing language.
    lang = 'en'
    if args['es']:
        lang = 'es'
        es = gettext.translation('eam', localedir='locale', languages=['es'])
        es.install()

    # Processing number of classes.
    num_classes = constants.n_labels
    if args['--num-classes']:
        num_classes = int(args['--num-classes'])
        assert 0 < num_classes
        constants.set_n_labels(num_classes)

    # Processing memory size (columns)
    if args['--domain']:
        constants.domain = int(args['--domain'])
        assert 0 < constants.domain
    # Processing runpath.
    if args['--runpath']:
        constants.run_path = args['--runpath']

    prefix = constants.memory_parameters_prefix
    filename = constants.csv_filename(prefix)
    parameters = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
    exp_settings = constants.ExperimentSettings(params=parameters)
    print(f'Working directory: {constants.run_path}')
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
