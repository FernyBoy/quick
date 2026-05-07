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
    hdf5_fname = os.path.join(constants.data_path, constants.prep_hdf5_fname)

    # Run the one-time loading/balancing logic if HDF5 doesn't exist
    if not os.path.exists(hdf5_fname):
        dataset_path = os.path.join(constants.data_path, constants.dataset_name)
        total_size = _load_dataset(dataset_path, hdf5_fname)
    else:
        with h5py.File(hdf5_fname, 'r') as f:
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
        hdf5_fname,
        segments,
        categorical=categorical,
        batch_size=constants.batch_size,
        predict_only=predict_only,
    )


def _load_dataset(path, hdf5_fname):
    """Coordinates the creation of the balanced HDF5 dataset in two passes to minimize RAM usage."""
    # Scan files to find the minimum images per class without loading data
    class_info, minimum_images = _scan_dataset_metadata(path)

    # Pass 2: Create the HDF5 and fill it class-by-class
    _save_dataset_streamed(class_info, minimum_images, path, hdf5_fname)

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
    """Semi-randomizes the dataset by processing chunks from all classes in parallel."""
    total_size = len(class_info) * min_imgs
    # Adjust buffer_per_class based on your available RAM
    buffer_per_class = 10000

    # Pre-map all files to avoid overhead in the loop
    mmaps = [np.load(info['path'], mmap_mode='r') for info in class_info]

    print(f'Creating Semi-Shuffled HDF5 at {hdf5_full_name}...')
    with h5py.File(hdf5_full_name, 'w') as f:
        ds_images = f.create_dataset(
            'images',
            shape=(total_size, 28, 28),
            dtype='uint8',
            chunks=(constants.batch_size, 28, 28),
            compression='gzip',
        )
        ds_labels = f.create_dataset('labels', shape=(total_size,), dtype='int32')

        write_ptr = 0
        for start in range(0, min_imgs, buffer_per_class):
            end = min(start + buffer_per_class, min_imgs)
            actual_chunk_size = end - start

            chunk_images = []
            chunk_labels = []

            for i, m in enumerate(mmaps):
                # Extract slice from memory-mapped file
                imgs = m[start:end].reshape(-1, 28, 28).astype('uint8')
                chunk_images.append(imgs)
                chunk_labels.append(np.full(actual_chunk_size, i, dtype='int32'))

            # Combine and shuffle this specific 'super-chunk' in RAM
            combined_imgs = np.concatenate(chunk_images, axis=0)
            combined_lbls = np.concatenate(chunk_labels, axis=0)

            shuffler = np.random.permutation(len(combined_lbls))

            # Write sequentially
            n_to_write = len(shuffler)
            ds_images[write_ptr : write_ptr + n_to_write] = combined_imgs[shuffler]
            ds_labels[write_ptr : write_ptr + n_to_write] = combined_lbls[shuffler]

            write_ptr += n_to_write
            print(f'Processed {write_ptr}/{total_size} images...')

    print('HDF5 creation complete.')


def _shuffle_dataset(data, labels):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
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
        predict_only=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hdf5_path = hdf5_path
        self.segments = segments
        self.categorical = categorical
        self.batch_size = batch_size
        self.predict_only = predict_only
        self.total_samples = sum(end - start for start, end in self.segments)
        self.data_file = None
        self.on_epoch_end()

    @property
    def num_batches(self):
        return self.__len__()

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
        data, labels = self._get_data_from_h5(start, count)
        data = data.astype('float32') / 255.0

        if self.predict_only:
            return data  # Just return the images for prediction
        # Categorical Conversion (Issue #1)
        if self.categorical:
            # Converts integer labels to one-hot vectors
            labels = keras.utils.to_categorical(
                labels, num_classes=constants.network_labels
            )
        return data, {'classifier': labels, 'decoder': data}

    def _get_data_from_h5(self, start, count):
        """Helper to fetch a slice by jumping through the ranges."""
        remaining = count
        current = start
        results_data = []
        results_labels = []

        for s_start, s_end in self.segments:
            range_len = s_end - s_start

            if current < range_len:
                # How much can we take from this specific range?
                take = min(remaining, range_len - current)

                # Physical slice in the H5 file
                h5_start = s_start + current
                h5_end = h5_start + take

                results_data.append(self.data_file['images'][h5_start:h5_end])
                if not self.predict_only:
                    results_labels.append(self.data_file['labels'][h5_start:h5_end])

                remaining -= take
                current = 0  # Next range starts from its beginning
            else:
                # Skips this range entirely
                current -= range_len

            if remaining <= 0:
                break

        # Combine the chunks (only happens at the 'gap' boundary)
        data = np.concatenate(results_data, axis=0)
        labels = (
            np.concatenate(results_labels, axis=0) if not self.predict_only else None
        )
        return data, labels

    def get_all_labels(self):
        """Efficiently retrieves all labels without loading a single image."""
        if self.data_file is None:
            self.data_file = h5py.File(self.hdf5_path, 'r', swmr=True)
        label_chunks = []
        for start, end in self.segments:
            # We slice ONLY the labels dataset
            label_chunks.append(self.data_file['labels'][start:end])
        all_labels = np.concatenate(label_chunks, axis=0)
        if self.categorical:
            return keras.utils.to_categorical(
                all_labels, num_classes=constants.network_labels
            )

        return all_labels
