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

import csv
import os

# os.environ['CUDA_VISIBLE_DEVICES']='0'
import re
import sys
import numpy as np

data_path = 'data/quick'
run_prefix = 'runs'
run_path = run_prefix
n_labels_path = None
idx_digits = 3
prep_hdf5_fname = 'prep_dataset.h5'
prep_names_fname = 'prep_names.csv'

image_path = 'images'
testing_path = 'test'
memories_path = 'memories'
dreams_path = 'dreams'

data_prefix = 'data'
labels_prefix = 'labels'
features_prefix = 'features'
memories_prefix = 'memories'
mem_conf_prefix = 'mem_confrix'
model_prefix = 'model'
recognition_prefix = 'recognition'
weights_prefix = 'weights'
classification_prefix = 'classification'
stats_prefix = 'model_stats'
learn_params_prefix = 'learn_params'
memory_parameters_prefix = 'mem_params'
chosen_prefix = 'chosen'
graph_prefix = 'graph'

balanced_data = 'balanced'
seed_data = 'seed'
learning_data_seed = 'seed_balanced'
learning_data_learned = 'learned'

# Categories suffixes.
original_suffix = '-original'
training_suffix = '-training'
filling_suffix = '-filling'
testing_suffix = '-testing'
memories_suffix = '-memories'

# Model suffixes.
encoder_suffix = '-encoder'
classifier_suffix = '-classifier'
decoder_suffix = '-decoder'
memory_suffix = '-memory'

data_suffix = '_X'
labels_suffix = '_Y'
matrix_suffix = '-confrix'

agreed_suffix = '-agr'
original_suffix = '-ori'
amsystem_suffix = '-ams'
nnetwork_suffix = '-rnn'
learning_suffixes = [
    [original_suffix],
    [agreed_suffix],
    [amsystem_suffix],
    [nnetwork_suffix],
    [original_suffix, amsystem_suffix],
]

# Number of columns in memory, which it is also the dimension of the latent representation
# of the neural networks. It must be a power of two greater than 4.
domain = 256
n_folds = 1
n_jobs = 1
# Batch size is set considering over 7 million elements of data and
# 2 L4 GPUs with 24 GB of RAM each.
# It should be a power of two.
batch_size = 2048
dreaming_cycles = 6

iota_default = 0.0
kappa_default = 0.0
xi_default = 0.0
sigma_default = 0.25
params_defaults = [iota_default, kappa_default, xi_default, sigma_default]
iota_idx = 0
kappa_idx = 1
xi_idx = 2
sigma_idx = 3

nn_training_percent = 0.50
nn_validating_percent = 0.20
nn_testing_percent = 0.10
am_filling_percent = 0.20
am_testing_percent = nn_testing_percent

# The number of classes used for training the neural networks, and for testing
# the memory system. The first one must be a power of two (8 at least) while
# the second must be a pair number, because in the negation experiment only
# half of the classes are stored in the memory.
network_labels = 64
memory_labels = 64
all_memory_labels = range(memory_labels)


def set_memory_labels(num_classes):
    global memory_labels, all_memory_labels
    if (num_classes < 2) or (num_classes > memory_labels):
        # Only a number of classes lower than or equal to the default n_labels is allowed.
        raise ValueError(
            f'The number of classes must be between 2 and ({memory_labels}).'
        )
    memory_labels = num_classes
    all_memory_labels = list(range(memory_labels))


label_formats = [
    'r:v',
    'y--d',
    'g-.4',
    'y-.3',
    'k-.8',
    'y--^',
    'c-..',
    'm:*',
    'c-1',
    'b-p',
    'm-.D',
    'c:D',
    'r--s',
    'g:d',
    'm:+',
    'y-._',
    'm:_',
    'y--h',
    'g--*',
    'm:_',
    'g-_',
    'm:d',
]

precision_idx = 0
recall_idx = 1
accuracy_idx = 2
entropy_idx = 3
no_response_idx = 4
no_mis_response_idx = 5
correct_response_idx = 6
incorrect_response_idx = 7
correct_mis_response_idx = 8
incorrect_mis_response_idx = 9
n_behaviours = 10

response_behaviours = {
    correct_response_idx: 'Correct Response',
    incorrect_response_idx: 'Incorrect Response',
    no_response_idx: 'No Response',
    correct_mis_response_idx: 'Correct Misresponse',
    incorrect_mis_response_idx: 'Incorrect Misresponse',
    no_mis_response_idx: 'No Misresponse',
}

response_colors = {
    correct_response_idx: 'green',
    incorrect_response_idx: 'orange',
    no_response_idx: 'blue',
    correct_mis_response_idx: 'red',
    incorrect_mis_response_idx: 'purple',
    no_mis_response_idx: 'olive',
}
memory_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
memory_fills = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
n_best_memory_sizes = 2

use_percentiles = False
minimum_percentile = 0.5
maximum_percentile = 99.5


class ExperimentSettings:
    def __init__(
        self,
        *,
        params=None,
        iota=None,
        kappa=None,
        xi=None,
        sigma=None,
        exp_number=None,
    ):
        if params is None:
            self.mem_params = params_defaults
        else:
            # If not None, it must be a one dimensional array.
            assert isinstance(params, np.ndarray)
            assert params.ndim == 1
            # The dimension must have four elements
            # iota, kappa, xi, sigma
            shape = params.shape
            assert shape[0] == 4
            self.mem_params = params
        if iota is not None:
            self.mem_params[iota_idx] = iota
        if kappa is not None:
            self.mem_params[kappa_idx] = kappa
        if xi is not None:
            self.mem_params[xi_idx] = xi
        if sigma is not None:
            self.mem_params[sigma_idx] = sigma
        self.experiment_number = exp_number if exp_number is not None else 0

    @property
    def xi(self):
        return self.mem_params[xi_idx]

    @property
    def iota(self):
        return self.mem_params[iota_idx]

    @property
    def kappa(self):
        return self.mem_params[kappa_idx]

    @property
    def sigma(self):
        return self.mem_params[sigma_idx]

    def __str__(self):
        return f'ExperimentSettings(iota={self.iota}, kappa={self.kappa}, xi={self.xi}, sigma={self.sigma})'


def int_suffix(n, prefix=None):
    prefix = '' if prefix is None else '-' + prefix + '_'
    return prefix + str(n).zfill(3)


def float_suffix(x, prefix=None):
    prefix = '' if prefix is None else '-' + prefix + '_'
    return prefix + f'{x:.2f}'


def numeric_suffix(prefix, value):
    return '-' + prefix + '_' + str(value).zfill(3)


def exp_number_suffix(es):
    return '' if es.experiment_number == 0 else int_suffix(es.experiment_number, 'exp')


def fold_suffix(fold):
    return '' if fold is None else int_suffix(fold, 'fld')


def msize_suffix(msize):
    return int_suffix(msize, 'msz')


def fill_suffix(fill):
    return int_suffix(fill, 'fil')


def get_name_w_suffix(prefix):
    suffix = ''
    return prefix + suffix


def get_full_name(name, es=None):
    if es is None:
        return name
    suffix = exp_number_suffix(es)
    return name + suffix


# Currently, names include nothing about experiment settings.
def model_name(es=None):
    return model_prefix


def stats_model_name(es=None):
    return stats_prefix


def features_name(es=None):
    return features_prefix


def labels_name(es=None):
    return labels_prefix


def mem_params_name(es=None):
    return memory_parameters_prefix


def data_name(es=None):
    return data_prefix


def memories_name(es=None):
    return memories_prefix


def recognition_name(es=None):
    return recognition_prefix


def weights_name(es=None):
    return weights_prefix


def classification_name(es=None):
    return classification_prefix


def learn_params_name(es=None):
    return learn_params_prefix


def graph_name(es=None):
    return graph_prefix


def dirname(path):
    match = re.search('[^/]*$', path)
    if match is None:
        return path
    tuple = os.path.splitext(match.group(0))
    return os.path.dirname(path) if tuple[1] else path


def create_directory(path):
    try:
        os.makedirs(path)
        print(f'Directory {path} created.')
    except FileExistsError:
        print(f'Directory {path} already exists.')


def filename(name, es=None, fold=None, extension='', sub_dir=''):
    """Returns a file name in run_path directory with a given extension and an index"""
    # Create target directory & all intermediate directories if don't exists
    if sub_dir is None:
        sub_dir = ''
    try:
        dir_path = os.path.join(run_path, sub_dir)
        os.makedirs(dir_path)
        print(f'Directory {dir_path} created ')
    except FileExistsError:
        pass
    return os.path.join(
        dir_path, get_full_name(name, es) + fold_suffix(fold) + extension
    )


def csv_filename(name_prefix, es=None, fold=None, sub_dir=None):
    return filename(name_prefix, es, fold, '.csv', sub_dir)


def data_filename(name_prefix, es=None, fold=None, sub_dir=None):
    return filename(name_prefix, es, fold, '.npy', sub_dir)


def figure_filename(name_prefix, es, fold=None, sub_dir=None):
    return filename(name_prefix, es, fold, '.svg', sub_dir)


def shared_data_filename(name_prefix, fold=None):
    """Returns a file name for an INPUT npy file, always in the main run_path."""
    return data_filename(name_prefix, fold=fold)


def json_filename(name_prefix, es):
    return filename(name_prefix, es, extension='.json')


# region Functions for naming files for storing neural networks.
# Names of this files do not depend on experimental settings.


def model_filename(name_prefix, fold):
    # This function will always point to the main runs directory
    return os.path.join(run_path, get_full_name(name_prefix) + fold_suffix(fold))


def encoder_filename(name_prefix, fold):
    return model_filename(name_prefix + encoder_suffix, fold) + '.keras'


def classifier_filename(name_prefix, fold):
    return model_filename(name_prefix + classifier_suffix, fold) + '.keras'


def decoder_filename(name_prefix, fold):
    return model_filename(name_prefix + decoder_suffix, fold) + '.keras'


# endregion Functions for naming files for storing neural networks.


def image_filename(dirname, idx, label, suffix='', es=None, fold=None):
    """Provides a file name for an image that it is either the input data or
    the output of the memory system"""
    name_prefix = (
        image_path
        + '/'
        + dirname
        + '/'
        + str(label).zfill(3)
        + '_'
        + str(idx).zfill(5)
        + suffix
    )
    return filename(name_prefix, es, fold, extension='.png')


def testing_image_filename(path, idx, label, es, fold):
    return image_filename(path, idx, label, original_suffix, es, fold)


def prod_testing_image_filename(dir, idx, label, es, fold):
    return image_filename(dir, idx, label, testing_suffix, es, fold)


def memory_image_filename(dir, idx, label, es, fold):
    return image_filename(dir, idx, label, memory_suffix, es, fold)


def print_csv(data):
    writer = csv.writer(sys.stdout)
    if np.ndim(data) == 1:
        writer.writerow(data)
    else:
        writer.writerows(data)


def print_warning(*s):
    print('WARNING:', *s, file=sys.stderr)


def print_error(*s):
    print('ERROR:', *s, file=sys.stderr)


def print_counter(n, every, step=1, symbol='.', prefix=''):
    if n == 0:
        return
    e = n % every
    s = n % step
    if (e != 0) and (s != 0):
        return
    counter = symbol
    if e == 0:
        counter = ' ' + prefix + str(n) + ' '
    print(counter, end='', flush=True)
