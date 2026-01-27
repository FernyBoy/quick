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
import keras
from keras.utils import Sequence
import h5py
import constants

# This code is an abstraction for the Draw Quick! dataset,
columns = 28
rows = 28

# Default batch size for generators
default_batch_size = 2048

_TRAINING_SEGMENT = 0
_VALIDATING_SEGMENT = 1
_FILLING_SEGMENT = 2
_TESTING_SEGMENT = 3


def get_training(
    fold,
    categorical=False,
    batch_size=default_batch_size,
    shuffle=True,
    predict_only=False,
):
    return _get_segment(
        _TRAINING_SEGMENT,
        fold,
        categorical,
        batch_size=batch_size,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def get_validating(
    fold,
    categorical=False,
    batch_size=default_batch_size,
    shuffle=True,
    predict_only=False,
):
    return _get_segment(
        _VALIDATING_SEGMENT,
        fold,
        categorical,
        batch_size=batch_size,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def get_filling(fold, batch_size=default_batch_size, shuffle=True, predict_only=False):
    return _get_segment(
        _FILLING_SEGMENT,
        fold,
        batch_size=batch_size,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def get_testing(
    fold,
    categorical=False,
    batch_size=default_batch_size,
    shuffle=True,
    predict_only=False,
):
    return _get_segment(
        _TESTING_SEGMENT,
        fold,
        categorical=categorical,
        batch_size=batch_size,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def _get_segment(
    segment,
    fold,
    categorical=False,
    batch_size=default_batch_size,
    shuffle=True,
    predict_only=False,
):
    hdf5_path = os.path.join(constants.data_path, constants.prep_hdf5_fname)

    # Run the one-time loading/balancing logic if HDF5 doesn't exist
    if not os.path.exists(hdf5_path):
        total_size = _load_dataset(constants.data_path)

    # Use your existing fold logic to calculate start/end points
    # (Simplified version of your logic here)
    training_size = int(total_size * constants.nn_training_percent)
    validating_size = int(total_size * constants.nn_validating_percent)
    filling_size = int(total_size * constants.am_filling_percent)
    testing_size = int(total_size * constants.nn_testing_percent)
    step = int(total_size / constants.n_folds)
    i = fold * step
    j = i + training_size
    k = j + validating_size
    m = k + filling_size
    n = m + testing_size
    j = j % total_size
    k = k % total_size
    m = m % total_size
    n = n % total_size
    p, q = None, None
    if segment == _TRAINING_SEGMENT:
        p, q = i, j
    elif segment == _VALIDATING_SEGMENT:
        p, q = j, k
    elif segment == _FILLING_SEGMENT:
        p, q = k, m
    elif segment == _TESTING_SEGMENT:
        p, q = m, n
    # Create the index range for this segment
    segment_indices = np.arange(p, q)

    return QuickDrawGenerator(
        hdf5_path,
        segment_indices,
        categorical=categorical,
        batch_size=batch_size,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def _load_dataset(path):
    data, labels = _load_quickdraw(path)
    _save_dataset_as_hdf5(data, labels, path)
    total_size = data.shape[0]
    return total_size


def _save_dataset_as_hdf5(data, labels, path):
    """Saves the balanced, shuffled data into a permanent HDF5 container."""
    hdf5_fname = os.path.join(path, constants.prep_hdf5_fname)
    print(f'Creating HDF5 dataset at {hdf5_fname}...')

    with h5py.File(hdf5_fname, 'w') as f:
        # We store as uint8 (0-255) to save disk space (7M images = ~5.5GB)
        # If we stored as float32, it would be ~22GB!
        f.create_dataset('images', data=data.astype('uint8'), compression='gzip')
        f.create_dataset('labels', data=labels.astype('int32'))
    print('HDF5 creation complete.')


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
    return shuffle_dataset(data, labels)


def shuffle_dataset(data, labels):
    print('Shuffling the dataset before storing it...')
    # 1. Create an array of indices [0, 1, 2, ..., N-1]
    indices = np.arange(data.shape[0])

    # 2. Shuffle the indices in-place (very fast, low memory)
    np.random.shuffle(indices)

    print('Reordering arrays...')
    # 3. Use 'fancy indexing' to reorder both arrays in one go
    # This creates a new shuffled array.
    # For 7M images (uint8), this uses ~5.5GB of RAM temporarily.
    data = data[indices]
    labels = labels[indices]
    return data, labels


class QuickDrawGenerator(Sequence):
    def __init__(
        self,
        hdf5_path,
        indices,
        categorical=False,
        batch_size=2048,
        shuffle=True,
        predict_only=False,
    ):
        self.hdf5_path = hdf5_path
        self.indices = indices  # These are the indices for the specific fold/segment
        self.categorical = categorical
        self.batch_size = batch_size
        self.predict_only = predict_only
        # If predicting, we MUST NOT shuffle to keep track of which embedding belongs to which image
        self.shuffle = shuffle if not predict_only else False
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        # Extract the specific indices for this batch
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        # HDF5 performs better with sorted index access
        sort_map = np.argsort(batch_indices)
        rev_map = np.argsort(sort_map)
        sorted_indices = batch_indices[sort_map]

        with h5py.File(self.hdf5_path, 'r') as f:
            x = f['images'][sorted_indices]
            # Labels are retrieved only if not in predict-only mode
            if not self.predict_only:
                y = f['labels'][sorted_indices]
        # Reshape to (Batch, 28, 28, 1), normalize to [0, 1], and restore original order
        x = x[rev_map].reshape(-1, 28, 28, 1).astype('float32') / 255.0

        if self.predict_only:
            return x  # Just return the images for prediction
        y = y[rev_map]

        # 2. Categorical Conversion (Issue #1)
        if self.categorical:
            # Converts integer labels to one-hot vectors
            y = keras.utils.to_categorical(y, num_classes=self.num_classes)

        # Match your multi-head requirement
        return x, {'classifier': y, 'decoder': x}
