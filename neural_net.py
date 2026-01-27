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
import h5py
import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Dropout,
    Dense,
    Flatten,
    Reshape,
    Conv2DTranspose,
    BatchNormalization,
    LayerNormalization,
    SpatialDropout2D,
)
from keras.callbacks import EarlyStopping
import constants
import dataset

epochs = 300
patience = 10
truly_training_percentage = 0.80
# Batch size and the number of workers are adjusted to having 2 L4 GPUs.
num_workers = 12  # Number of CPU cores for data prep
batch_size = 2048


def conv_block(entry, layers, filters, dropout, first_block=False):
    conv = None
    for i in range(layers):
        if first_block:
            conv = Conv2D(
                kernel_size=3,
                padding='same',
                activation='relu',
                filters=filters,
            )(entry)
            first_block = False
        else:
            conv = Conv2D(
                kernel_size=3, padding='same', activation='relu', filters=filters
            )(entry)
        entry = BatchNormalization()(conv)
    pool = MaxPool2D(pool_size=2, strides=2, padding='same')(entry)
    drop = SpatialDropout2D(dropout)(pool)
    return drop


# The number of layers defined in get_encoder.
encoder_nlayers = 40


def get_encoder(domain):
    dropout = 0.5
    input_data = Input(shape=(dataset.columns, dataset.rows, 1))
    filters = domain // 16
    output = conv_block(input_data, 2, filters, dropout, first_block=True)
    filters *= 2
    dropout -= 0.05
    output = conv_block(output, 2, filters, dropout)
    filters *= 2
    dropout -= 0.05
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout -= 0.05
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout -= 0.05
    output = conv_block(output, 3, filters, dropout)
    output = Flatten()(output)
    output = LayerNormalization(name='encoded')(output)
    return input_data, output


def get_decoder(domain):
    input_mem = Input(shape=(domain,))
    width = dataset.columns // 4
    filters = domain // 4
    dense = Dense(width * width * filters, activation='relu')(input_mem)
    output = Reshape((width, width, filters))(dense)
    dropout = 0.4
    for i in range(2):
        trans = Conv2DTranspose(
            kernel_size=3, strides=2, padding='same', activation='relu', filters=filters
        )(output)
        output = SpatialDropout2D(dropout)(trans)
        dropout /= 2.0
        filters = filters // (constants.domain // 32)
        output = BatchNormalization()(output)
    output = Conv2DTranspose(
        filters=filters, kernel_size=3, strides=1, activation='sigmoid', padding='same'
    )(output)
    return input_mem, output


# The number of layers defined in get_classifier.
classifier_nlayers = 6


def get_classifier(domain):
    input_mem = Input(shape=(domain,))
    dense = Dense(domain, activation='relu')(input_mem)
    drop = Dropout(0.4)(dense)
    dense = Dense(domain, activation='relu')(drop)
    drop = Dropout(0.4)(dense)
    classification = Dense(constants.n_labels, activation='softmax', name='classified')(
        drop
    )
    return input_mem, classification


def train_network(prefix, es):
    confusion_matrix = np.zeros((constants.n_labels, constants.n_labels))
    histories = []
    for fold in range(constants.n_folds):
        training_gen = dataset.get_training(fold, categorical=True)
        # No shuffling is needed for validation nor testing.
        validating_gen = dataset.get_validating(fold, categorical=True, shuffle=False)
        testing_gen = dataset.get_testing(fold, categorical=True, shuffle=False)
        predict_gen = dataset.get_testing(fold, predict_only=True)

        rmse = tf.keras.metrics.RootMeanSquaredError()
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            domain = constants.domain
            input_data = Input(shape=(dataset.columns, dataset.rows, 1))
            input_enc, output_enc = get_encoder(domain)
            input_class, output_class = get_classifier(domain)
            input_dec, output_dec = get_decoder(domain)

            encoder = Model(input_enc, output_enc, name='encoder')
            classifier = Model(input_class, output_class, name='classifier')
            decoder = Model(input_dec, output_dec, name='decoder')
            encoded = encoder(input_data)
            decoded = decoder(encoded)
            classified = classifier(encoded)
            model = Model(
                inputs=input_data,
                outputs={'classifier': classified, 'decoder': decoded},
            )
            model.compile(
                loss=['categorical_crossentropy', 'mean_squared_error'],
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=2e-2
                ),  # Learning rate for a batch size of 2048
                metrics={'classifier': 'accuracy', 'decoder': rmse},
            )
            encoder.summary()
            classifier.summary()
            decoder.summary()
            model.summary()

            full_classifier = Model(
                inputs=input_data, outputs=classified, name='full_classifier'
            )
            # autoencoder = Model(inputs=input_data, outputs=decoded, name='autoencoder')

        early_stopping = EarlyStopping(
            monitor='val_classifier_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='min',
            verbose=1,
        )
        history = model.fit(
            training_gen,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validating_gen,
            callbacks=[early_stopping],
            workers=num_workers,  # Use multiple CPU cores for data prep
            use_multiprocessing=True,  # True if your generator is thread-safe
            verbose=2,
        )
        histories.append(history)
        history = model.evaluate(testing_gen, return_dict=True)
        histories.append(history)
        predicted_labels = np.argmax(full_classifier.predict(predict_gen), axis=1)
        # Retrieve True Labels directly from HDF5 using generator indices
        with h5py.File(testing_gen.h5_path, 'r') as f:
            true_labels = f['labels'][testing_gen.indices]
        confusion_matrix += tf.math.confusion_matrix(
            true_labels,
            predicted_labels,
            num_classes=constants.n_labels,
        )
        encoder.save(constants.encoder_filename(prefix, es, fold))
        decoder.save(constants.decoder_filename(prefix, es, fold))
        classifier.save(constants.classifier_filename(prefix, es, fold))
        prediction_prefix = constants.classification_name(es)
        prediction_filename = constants.data_filename(prediction_prefix, es, fold)
        np.save(prediction_filename, predicted_labels)
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1, 1)
    return histories, confusion_matrix / totals


def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix, es):
    for fold in range(constants.n_folds):
        # Load the encoder
        filename = constants.encoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(filename)

        # 1. Get Generators (which replace the raw data arrays)
        # We set predict_only=True so the generator returns ONLY images for model.predict
        train_gen = dataset.get_training(fold, predict_only=True)
        fill_gen = dataset.get_filling(fold, predict_only=True)
        test_gen = dataset.get_testing(fold, predict_only=True)
        settings = [
            (train_gen, constants.training_suffix),
            (fill_gen, constants.filling_suffix),
            (test_gen, constants.testing_suffix),
        ]

        for gen, suffix in settings:
            # 2. Parallel Prediction on Dual L4s
            # Keras handles the distribution if you wrap this in a strategy scope
            # or simply rely on its internal optimization for generators.
            print(f'Generating features for {suffix}...')
            features = model.predict(
                gen,
                workers=num_workers,
                use_multiprocessing=True,
                verbose=1,
            )

            # 3. Retrieve Labels and Raw Data for Saving
            # Since we can't hold all 7M images in RAM easily, we pull them
            # from the H5 file using the indices stored in the generator.
            with h5py.File(gen.h5_path, 'r') as f:
                # We use the generator's indices to ensure the order matches the features
                indices = gen.indices
                data = f['images'][indices]
                labels = f['labels'][indices]

            # 4. Save to .npy (as per your original requirement)
            data_filename = constants.data_filename(data_prefix + suffix, es, fold)
            features_filename = constants.data_filename(
                features_prefix + suffix, es, fold
            )
            labels_filename = constants.data_filename(labels_prefix + suffix, es, fold)

            print('Saving data, features and labels ...')
            np.save(data_filename, data)
            np.save(features_filename, features)
            np.save(labels_filename, labels)
