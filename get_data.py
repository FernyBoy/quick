# Copyright 2026 Luis Alberto Pineda, Rafael Morales
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Get QuickDraw dataset from Google Cloud Storage.

Usage:
    get_data -h | --help
    get_data <num_clases>
"""

import os
import random
from docopt import docopt
from google.cloud import storage
import constants

seed = 42
bucket_name = 'quickdraw_dataset'
source_blob_prefix = 'full/numpy_bitmap/'
destination_folder = constants.data_path


def download_public_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a public blob from the bucket."""

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print('done!')


def download_files(blob_names):
    for blob_name in blob_names:
        destination_file_name = os.path.join(
            destination_folder, os.path.basename(blob_name)
        )
        print(f'Downloading {blob_name} to {destination_file_name}...', end=' ')
        download_public_file(bucket_name, blob_name, destination_file_name)


def get_blob_names(bucket_name, prefix):
    storage_client = storage.Client.create_anonymous_client()
    # List all blobs and store their names in a list
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # Using a list comprehension to extract just the file paths
    file_names = [blob.name for blob in blobs]
    return file_names


if __name__ == '__main__':
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    args = docopt(__doc__)
    num_classes = int(args['<num_clases>'])
    if num_classes <= 0:
        raise ValueError('<num_clases> must be a positive integer.')

    blob_names = get_blob_names(bucket_name, source_blob_prefix)
    chosen_blob_names = random.sample(blob_names, num_classes)
    download_files(chosen_blob_names)
