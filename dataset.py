# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
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

import numpy as np
import os
import random
import constants
import keras

# This code is an abstraction for the Draw Quick! dataset,
columns = 28
rows = 28


def get_training(fold, categorical=False):
    return _get_segment(_TRAINING_SEGMENT, fold, categorical)


def get_filling(fold):
    return _get_segment(_FILLING_SEGMENT, fold)


def get_testing(fold, categorical=False, noised=False):
    return _get_segment(_TESTING_SEGMENT, fold, categorical, noised)


def _get_segment(segment, fold, categorical=False, noised=False):
    if (
        (_get_segment.data is None)
        or (_get_segment.noised is None)
        or (_get_segment.labels is None)
    ):
        _get_segment.data, _get_segment.noised, _get_segment.labels = _load_dataset(
            constants.data_path
        )
    total = len(_get_segment.labels)
    training = total * constants.nn_training_percent
    filling = total * constants.am_filling_percent
    testing = total * constants.am_testing_percent
    step = total / constants.n_folds
    i = fold * step
    j = i + training
    k = j + filling
    l = k + testing
    i = int(i)
    j = int(j) % total
    k = int(k) % total
    l = int(l) % total
    n, m = None, None
    if segment == _TRAINING_SEGMENT:
        n, m = i, j
    elif segment == _FILLING_SEGMENT:
        n, m = j, k
    elif segment == _TESTING_SEGMENT:
        n, m = k, l

    data = (
        constants.get_data_in_range(_get_segment.noised, n, m)
        if noised
        else constants.get_data_in_range(_get_segment.data, n, m)
    )
    labels = constants.get_data_in_range(_get_segment.labels, n, m)

    num_classes = np.unique_values(labels).shape[0]
    if num_classes < constants.n_labels:
        constants.print_error(
            f'Only {num_classes} classes found instead of {constants.n_labels}.'
        )
        exit(1)
    elif num_classes > constants.n_labels:
        constants.print_warning(
            f'Found {num_classes} classes, but only {constants.n_labels} are used.'
        )
        mask = labels < constants.n_labels
        labels = labels[mask]
        data = data[mask]

    # Convert labels to one-hot encoding
    if categorical:
        labels = keras.utils.to_categorical(labels, num_classes=constants.n_labels)
    return data, labels


_get_segment.data = None
_get_segment.noised = None
_get_segment.labels = None


def noised(data, percent):
    print(f'Adding {percent}% noise to data.')
    copy = np.zeros(data.shape, dtype=float)
    n = 0
    for i in range(len(copy)):
        copy[i] = _noised(data[i], percent)
        n += 1
        constants.print_counter(n, 10000, step=100)
    return copy


def _noised(image, percent):
    copy = np.array([row[:] for row in image])
    total = round(columns * rows * percent / 100.0)
    noised = []
    while len(noised) < total:
        i = random.randrange(rows)
        j = random.randrange(columns)
        if (i, j) in noised:
            continue
        value = random.random()
        copy[i, j] = value
        noised.append((i, j))
    return copy


_TRAINING_SEGMENT = 0
_FILLING_SEGMENT = 1
_TESTING_SEGMENT = 2


def _load_dataset(path):
    data, noised_data, labels = _preprocessed_dataset(path)
    if (data is None) or (noised_data is None) or (labels is None):
        data, labels = _load_quickdraw(path)
        data = data.astype(float) / 255.0
        noised_data = noised(data, constants.noise_percent)
        data, noised_data, labels = _shuffle(data, noised_data, labels)
        _save_dataset(data, noised_data, labels, path)
    return data, noised_data, labels


def _preprocessed_dataset(path):
    data_fname = os.path.join(path, constants.prep_data_fname)
    noised_fname = os.path.join(path, constants.pred_noised_data_fname)
    labels_fname = os.path.join(path, constants.prep_labels_fname)
    data = None
    noised = None
    labels = None
    try:
        data = np.load(data_fname)
        noised = np.load(noised_fname)
        labels = np.load(labels_fname).astype('int')
        print('Preprocessed dataset exists, so it is used.')
    except FileNotFoundError:
        print('Preprocessed dataset does not exist, so it will be created.')
    except Exception as e:
        print(f'Error loading preprocessed dataset: {e}')
        exit(1)
    return data, noised, labels


def _save_dataset(data, noised, labels, path):
    print('Saving preprocessed dataset')
    data_fname = os.path.join(path, constants.prep_data_fname)
    noised_fname = os.path.join(path, constants.pred_noised_data_fname)
    labels_fname = os.path.join(path, constants.prep_labels_fname)
    np.save(data_fname, data)
    np.save(noised_fname, noised)
    np.save(labels_fname, labels)


def _load_quickdraw(path):
    """
    Loads all .npy QuickDraw files in a directory and assigns numeric labels.
    Returns:
        data: ndarray of shape (N, 28, 28)
        labels: ndarray of integers of shape (N,)
    """
    print('Loading QuickDraw .npy files...')
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
    random.shuffle(files)
    if len(files) < constants.n_labels:
        constants.print_error(
            f'Only {len(files)} classes found instead of at least {constants.n_labels}.'
        )
        exit(1)
    # Only data that is going to be used is included in the dataset.
    files = files[: constants.n_labels]
    data_list = []
    labels_list = []
    label_dict = {}
    minimum_images = -1
    temp_data_list = []
    temp_labels_list = []

    for label_index, filename in enumerate(files):
        full_path = os.path.join(path, filename)
        class_name = filename.replace('full_numpy_bitmap_', '').replace('.npy', '')
        label_dict[label_index] = class_name

        print(f'Loading {class_name} from {full_path}...')
        images = np.load(full_path)
        if minimum_images == -1:
            minimum_images = images.shape[0]
        elif images.shape[0] < minimum_images:
            minimum_images = images.shape[0]
        images = images.astype(float).reshape(-1, 28, 28)

        temp_data_list.append(images)
        temp_labels_list.append(np.full(len(images), label_index, dtype=int))

    print(f'Balancing the dataset to {minimum_images} per class')
    for data, labels in zip(temp_data_list, temp_labels_list):
        data_list.append(data[:minimum_images])
        labels_list.append(labels[:minimum_images])

    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f'Loaded a total of {data.shape[0]} images of {len(label_dict)} classes.')
    return data, labels


def _shuffle(data, noised, labels):
    print('Shuffling data and labels')
    tuples = [(data[i], noised[i], labels[i]) for i in range(len(labels))]
    random.shuffle(tuples)
    data = np.array([p[0] for p in tuples])
    noised = np.array([p[1] for p in tuples])
    labels = np.array([p[2] for p in tuples], dtype=int)
    return data, noised, labels
