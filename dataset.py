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

import time
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

_TRAINING_SEGMENT = 0
_VALIDATING_SEGMENT = 1
_FILLING_SEGMENT = 2
_TESTING_SEGMENT = 3


def get_training(
    fold,
    categorical=False,
    shuffle=True,
    predict_only=False,
):
    return _get_segment(
        _TRAINING_SEGMENT,
        fold,
        categorical,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def get_validating(
    fold,
    categorical=False,
    shuffle=True,
    predict_only=False,
):
    return _get_segment(
        _VALIDATING_SEGMENT,
        fold,
        categorical,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def get_filling(fold, shuffle=True, predict_only=False):
    return _get_segment(
        _FILLING_SEGMENT,
        fold,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def get_testing(
    fold,
    categorical=False,
    shuffle=True,
    predict_only=False,
):
    return _get_segment(
        _TESTING_SEGMENT,
        fold,
        categorical=categorical,
        shuffle=shuffle,
        predict_only=predict_only,
    )


def _get_segment(
    segment,
    fold,
    categorical=False,
    shuffle=True,
    predict_only=False,
):
    hdf5_path = os.path.join(constants.data_path, constants.prep_hdf5_fname)

    # Run the one-time loading/balancing logic if HDF5 doesn't exist
    if not os.path.exists(hdf5_path):
        total_size = _load_dataset(constants.data_path)
    else:
        with h5py.File(hdf5_path, 'r') as f:
            total_size = f['labels'].shape[0]

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
    if segment == _TRAINING_SEGMENT:
        p, q = i, j
    elif segment == _VALIDATING_SEGMENT:
        p, q = j, k
    elif segment == _FILLING_SEGMENT:
        p, q = k, m
    elif segment == _TESTING_SEGMENT:
        p, q = m, n

    if p < q:
        segments = [(p, q)]
    else:
        segments = [(p, total_size), (0, q)]
    return QuickDrawGenerator(
        hdf5_path,
        segments,
        categorical=categorical,
        batch_size=constants.batch_size,
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
    data, labels = _shuffle_dataset(data, labels)
    hdf5_fname = os.path.join(path, constants.prep_hdf5_fname)
    print(f'Creating HDF5 dataset at {hdf5_fname}...')

    with h5py.File(hdf5_fname, 'w') as f:
        # We store as uint8 (0-255) to save disk space (7M images = ~5.5GB)
        # If we stored as float32, it would be ~22GB!
        f.create_dataset(
            'images',
            data=data.astype('uint8'),
            chunks=(constants.batch_size, 28, 28),  # Store data in batch-sized blocks
            compression='gzip',
        )
        f.create_dataset('labels', data=labels.astype('int32'))
    print('HDF5 creation complete.')


def _load_quickdraw(path):
    """
    Loads all .npy QuickDraw files in a directory and assigns numeric labels.
    Returns:
        data: ndarray of shape (N, 28, 28)
        labels: ndarray of integers of shape (N,)
    It saves the label mapping in CSV file.
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
    label_names = []
    minimum_images = -1
    temp_data_list = []
    temp_labels_list = []

    for label_index, filename in enumerate(files):
        full_path = os.path.join(path, filename)
        name = filename.replace('full_numpy_bitmap_', '').replace('.npy', '')
        label_names.append(name)

        print(f'Loading {name} from {full_path}...')
        images = np.load(full_path)
        if minimum_images == -1:
            minimum_images = images.shape[0]
        elif images.shape[0] < minimum_images:
            minimum_images = images.shape[0]
        images = images.astype(float).reshape(-1, 28, 28)

        temp_data_list.append(images)
        temp_labels_list.append(np.full(len(images), label_index, dtype=int))

    csv_path = os.path.join(constants.data_path, constants.prep_names_fname)
    with open(csv_path, 'w') as file:
        file.write('\n'.join(label_names))

    print(f'Balancing the dataset to {minimum_images} per class')
    for data, labels in zip(temp_data_list, temp_labels_list):
        data_list.append(data[:minimum_images])
        labels_list.append(labels[:minimum_images])

    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f'Loaded a total of {data.shape[0]} images of {len(label_names)} classes.')
    return data, labels


def _shuffle_dataset(data, labels):
    # 1. Create an array of indices [0, 1, 2, ..., N-1]
    indices = np.arange(data.shape[0])

    # 2. Shuffle the indices in-place (very fast, low memory)
    np.random.shuffle(indices)

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
        segments,
        categorical=False,
        batch_size=2048,
        shuffle=True,
        predict_only=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hdf5_path = hdf5_path
        self.segments = segments
        self.categorical = categorical
        self.batch_size = batch_size
        self.shuffle = shuffle and not predict_only
        self.predict_only = predict_only
        self.total_samples = sum(end - start for start, end in self.segments)
        self.data_file = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.total_samples / self.batch_size))

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        # Lazy initialization
        if self.data_file is None:
            nbytes = (constants.batch_size // 2) ** 2 * (constants.batch_size // 4)
            self.data_file = h5py.File(
                self.hdf5_path, 'r', rdcc_nbytes=nbytes
            )  # 512MB Cache
        # Extract the specific indices for this batch
        start = idx * self.batch_size
        # Retrieves what remains if it is not a full batch
        count = min(self.batch_size, self.total_samples - start)
        x, y = self._get_data_from_h5(start, count)
        x = x.astype('float32') / 255.0

        if self.predict_only:
            return x  # Just return the images for prediction
        if self.shuffle:
            x, y = _shuffle_dataset(x, y)
        # 2. Categorical Conversion (Issue #1)
        if self.categorical:
            # Converts integer labels to one-hot vectors
            y = keras.utils.to_categorical(y, num_classes=constants.n_labels)
        return x, {'classifier': y, 'decoder': x}

    def _get_data_from_h5(self, start, count):
        """Helper to fetch a slice by jumping through the ranges."""
        remaining = count
        current = start
        results_x = []
        results_y = []

        for s_start, s_end in self.segments:
            range_len = s_end - s_start

            if current < range_len:
                # How much can we take from this specific range?
                take = min(remaining, range_len - current)

                # Physical slice in the H5 file
                h5_start = s_start + current
                h5_end = h5_start + take

                results_x.append(self.data_file['images'][h5_start:h5_end])
                if not self.predict_only:
                    results_y.append(self.data_file['labels'][h5_start:h5_end])

                remaining -= take
                current = 0  # Next range starts from its beginning
            else:
                # Skips this range entirely
                current -= range_len

            if remaining <= 0:
                break

        # Combine the chunks (only happens at the 'gap' boundary)
        x = np.concatenate(results_x, axis=0)
        y = np.concatenate(results_y, axis=0) if not self.predict_only else None
        return x, y

    def get_all_labels(self):
        """Efficiently retrieves all labels without loading a single image."""
        if self.data_file is None:
            self.data_file = h5py.File(self.hdf5_path, 'r', swmr=True)

        label_chunks = []
        for start, end in self.segments:
            # We slice ONLY the labels dataset
            label_chunks.append(self.data_file['labels'][start:end])

        all_labels = np.concatenate(label_chunks, axis=0)

        # No shuffling is performed here, as the order matters for evaluation
        if self.categorical:
            return keras.utils.to_categorical(
                all_labels, num_classes=constants.n_labels
            )

        return all_labels
