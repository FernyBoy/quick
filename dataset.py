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

_TRAINING_SEGMENT = 0
_VALIDATING_SEGMENT = 1
_FILLING_SEGMENT = 2
_TESTING_SEGMENT = 3


def get_training(
    fold,
    categorical=False,
    predict_only=False,
):
    return _get_segment(
        _TRAINING_SEGMENT,
        fold,
        categorical,
        predict_only=predict_only,
    )


def get_validating(
    fold,
    categorical=False,
    predict_only=False,
):
    return _get_segment(
        _VALIDATING_SEGMENT,
        fold,
        categorical,
        predict_only=predict_only,
    )


def get_filling(fold, predict_only=False):
    return _get_segment(
        _FILLING_SEGMENT,
        fold,
        predict_only=predict_only,
    )


def get_testing(
    fold,
    categorical=False,
    predict_only=False,
):
    return _get_segment(
        _TESTING_SEGMENT,
        fold,
        categorical=categorical,
        predict_only=predict_only,
    )


def _get_segment(
    segment,
    fold,
    categorical=False,
    predict_only=False,
):
    hdf5_full_name = os.path.join(constants.data_path, constants.prep_hdf5_fname)

    # Run the one-time loading/balancing logic if HDF5 doesn't exist
    if not os.path.exists(hdf5_full_name):
        dataset_path = os.path.join(constants.data_path, constants.dataset_name)
        total_size = _load_dataset(dataset_path, hdf5_full_name)
    else:
        with h5py.File(hdf5_full_name, 'r') as f:
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
        hdf5_full_name,
        segments,
        categorical=categorical,
        batch_size=constants.batch_size,
        predict_only=predict_only,
    )


def _load_dataset(dataset_path, hdf5_full_name):
    """Coordinates the creation of the balanced HDF5 dataset in two passes to minimize RAM usage."""
    # Scan files to find the minimum images per class without loading data
    class_info, minimum_images = _scan_dataset_metadata(dataset_path)

    # Pass 2: Create the HDF5 and fill it class-by-class
    _save_dataset_streamed(class_info, minimum_images, hdf5_full_name)

    total_size = len(class_info) * minimum_images
    return total_size


def _scan_dataset_metadata(path):
    """Determines the minimum images per class using memory-mapping."""
    print('Scanning QuickDraw metadata...')
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
    if len(files) < constants.network_labels:
        raise ValueError(
            f'Not enough classes found in {path}. '
            f'Expected at least {constants.network_labels}, found {len(files)}.'
        )
    random.shuffle(files)
    files = files[: constants.network_labels]

    class_info = []
    minimum_images = -1
    label_names = []

    for filename in files:
        full_path = os.path.join(path, filename)
        name = filename.replace('full_numpy_bitmap_', '').replace('.npy', '')

        # mmap_mode='r' allows us to see the shape without loading into RAM
        images = np.load(full_path, mmap_mode='r')
        count = images.shape[0]

        if minimum_images == -1 or count < minimum_images:
            minimum_images = count

        class_info.append({'name': name, 'path': full_path})
        label_names.append(name)

    # Save the label mapping to a CSV for later reference.
    csv_path = os.path.join(constants.data_path, constants.prep_names_fname)
    with open(csv_path, 'w') as file:
        file.write('\n'.join(label_names))

    print(f'Balancing dataset to {minimum_images} images per class.')
    return class_info, minimum_images


def _save_dataset_streamed(class_info, min_imgs, hdf5_full_name):
    """Writes sequentially to HDF5. Shuffling is handled by the Generator."""
    total_size = len(class_info) * min_imgs

    print(f'Creating Sequential HDF5 at {hdf5_full_name}...')
    with h5py.File(hdf5_full_name, 'w') as f:
        # Keep compression if you want, sequential writes handle it much better
        ds_images = f.create_dataset(
            'images',
            shape=(total_size, 28, 28),
            dtype='uint8',
            chunks=(constants.batch_size, 28, 28),
            compression='gzip',
        )
        ds_labels = f.create_dataset('labels', shape=(total_size,), dtype='int32')

        for i, info in enumerate(class_info):
            print(f'Streaming class {i}: {info["name"]}...')
            imgs = np.load(info['path'])[:min_imgs].reshape(-1, 28, 28).astype('uint8')
            lbls = np.full(min_imgs, i, dtype='int32')

            # WRITE SEQUENTIALLY: No random seeking
            start_idx = i * min_imgs
            end_idx = start_idx + min_imgs
            ds_images[start_idx:end_idx] = imgs
            ds_labels[start_idx:end_idx] = lbls

    # Save a global shuffle map once the file is done
    print('Generating and saving global shuffle map...')
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    np.save(os.path.join(constants.data_path, constants.prep_shuffled_map), indices)
    print('Streamed HDF5 creation complete.')


class QuickDrawGenerator(Sequence):
    def __init__(
        self,
        hdf5_path,
        segments,
        categorical=False,
        batch_size=2048,
        predict_only=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hdf5_path = hdf5_path
        self.segments = segments
        self.batch_size = batch_size
        self.predict_only = predict_only
        self.categorical = categorical

        # Load the global map we created during HDF5 creation
        map_path = os.path.join(os.path.dirname(hdf5_path), constants.prep_shuffled_map)
        self.global_map = np.load(map_path)

        # Filter the map to only include indices within our fold's segments
        fold_indices = []
        for start, end in self.segments:
            fold_indices.extend(range(start, end))
        self.my_indices = self.global_map[
            fold_indices
        ]  # The 'shuffled' truth for this fold

        self.total_samples = len(self.my_indices)
        self.data_file = None

    def __getitem__(self, idx):
        if self.data_file is None:
            self.data_file = h5py.File(self.hdf5_path, 'r')

        start = idx * self.batch_size
        end = min(start + self.batch_size, self.total_samples)

        # Get the specific 'shuffled' indices for this batch
        batch_indices = self.my_indices[start:end]

        # Fancy indexing on read is faster than on write
        # We sort them to help HDF5 read speed
        sort_idx = np.argsort(batch_indices)
        rev_sort_idx = np.argsort(sort_idx)

        data = self.data_file['images'][batch_indices[sort_idx]][rev_sort_idx]
        data = data.astype('float32') / 255.0

        if self.predict_only:
            return data

        labels = self.data_file['labels'][batch_indices[sort_idx]][rev_sort_idx]
        if self.categorical:
            labels = keras.utils.to_categorical(
                labels, num_classes=constants.network_labels
            )

        return data, {'classifier': labels, 'decoder': data}

    def _get_data_from_h5(self, start, count):
        """Fetches shuffled samples using the virtual index map."""
        if self.data_file is None:
            # swmr=True (Single Writer Multiple Reader) is faster for reading
            self.data_file = h5py.File(self.hdf5_path, 'r', swmr=True)

        # 1. Get the physical row numbers from our pre-shuffled map
        end = start + count
        batch_physical_indices = self.my_indices[start:end]

        # 2. Optimization: HDF5 reads are 10x faster if indices are sorted
        sort_idx = np.argsort(batch_physical_indices)
        rev_sort_idx = np.argsort(sort_idx)
        sorted_indices = batch_physical_indices[sort_idx]

        # 3. Pull from disk
        data = self.data_file['images'][sorted_indices]
        # Put them back into the shuffled order the model expects
        data = data[rev_sort_idx]

        labels = None
        if not self.predict_only:
            labels = self.data_file['labels'][sorted_indices]
            labels = labels[rev_sort_idx]

        return data, labels

    def get_all_labels(self):
        """Retrieves all labels for this fold in their shuffled order."""
        if self.data_file is None:
            self.data_file = h5py.File(self.hdf5_path, 'r', swmr=True)

        # We use the map to get physical locations, sort for speed, then unsort
        all_phys_indices = self.my_indices
        sort_idx = np.argsort(all_phys_indices)
        rev_sort_idx = np.argsort(sort_idx)

        # Pull only the labels column (very memory efficient)
        all_labels = self.data_file['labels'][all_phys_indices[sort_idx]]
        all_labels = all_labels[rev_sort_idx]

        if self.categorical:
            return keras.utils.to_categorical(
                all_labels, num_classes=constants.network_labels
            )

        return all_labels
