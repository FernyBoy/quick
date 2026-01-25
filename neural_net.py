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
from keras.utils import Sequence
from keras.callbacks import EarlyStopping
import constants
import dataset

batch_size = 100
epochs = 300
patience = 10
truly_training_percentage = 0.80


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
    dense = Dense(width * width * filters, activation='relu')(
        input_mem
    )
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
        training_data, training_labels = dataset.get_training(fold, categorical=True)
        testing_data, testing_labels = dataset.get_testing(fold, categorical=True)
        truly_training = int(len(training_labels) * truly_training_percentage)
        validation_data = training_data[truly_training:]
        validation_labels = training_labels[truly_training:]
        training_data = training_data[:truly_training]
        training_labels = training_labels[:truly_training]

        training_generator = DataGeneratorForTraining(
            training_data, training_labels, batch_size
        )
        validation_generator = DataGeneratorForTraining(
            validation_data, validation_labels, batch_size
        )
        testing_generator_for_training = DataGeneratorForTraining(
            testing_data, testing_labels, batch_size
        )
        testing_generator_for_predicting = DataGenerator(testing_data, batch_size)
        rmse = tf.keras.metrics.RootMeanSquaredError()
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_data = Input(shape=(dataset.columns, dataset.rows, 1))
            domain = constants.domain
            input_enc, encoded = get_encoder(domain)
            encoder = Model(input_enc, encoded, name='encoder')
            encoder.compile(optimizer='adam')
            encoder.summary()
            input_cla, classified = get_classifier(domain)
            classifier = Model(input_cla, classified, name='classifier')
            classifier.compile(
                loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
            )
            classifier.summary()
            input_dec, decoded = get_decoder(domain)
            decoder = Model(input_dec, decoded, name='decoder')
            decoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmse])
            decoder.summary()
            encoded = encoder(input_data)
            decoded = decoder(encoded)
            classified = classifier(encoded)
            full_classifier = Model(
                inputs=input_data, outputs=classified, name='full_classifier'
            )
            full_classifier.compile(
                optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
            )
            autoencoder = Model(inputs=input_data, outputs=decoded, name='autoencoder')
            autoencoder.compile(loss='huber', optimizer='adam', metrics=[rmse])

            model = Model(inputs=input_data, outputs=[classified, decoded])
            model.compile(
                loss=['categorical_crossentropy', 'mean_squared_error'],
                optimizer='adam',
                metrics={'classifier': 'accuracy', 'decoder': rmse},
            )
            model.summary()
        early_stopping = EarlyStopping(
            monitor='val_classifier_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='min',
            verbose=1,
        )
        history = model.fit(
            training_generator,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping],
            verbose=2,
        )
        histories.append(history)
        history = full_classifier.evaluate(
            testing_generator_for_training, return_dict=True
        )
        histories.append(history)
        predicted_labels = np.argmax(
            full_classifier.predict(testing_generator_for_predicting), axis=1
        )
        confusion_matrix += tf.math.confusion_matrix(
            np.argmax(testing_labels, axis=1),
            predicted_labels,
            num_classes=constants.n_labels,
        )
        history = autoencoder.evaluate(testing_data, testing_data, return_dict=True)
        histories.append(history)
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
    """Generate features for sound segments, corresponding to phonemes.

    Uses the previously trained neural networks for generating the features.
    """
    for fold in range(constants.n_folds):
        # Load de encoder
        filename = constants.encoder_filename(model_prefix, es, fold)
        model = tf.keras.models.load_model(filename)
        model.summary()

        training_data, training_labels = dataset.get_training(fold)
        filling_data, filling_labels = dataset.get_filling(fold)
        testing_data, testing_labels = dataset.get_testing(fold)
        settings = [
            (training_data, training_labels, constants.training_suffix),
            (filling_data, filling_labels, constants.filling_suffix),
            (testing_data, testing_labels, constants.testing_suffix),
        ]
        for s in settings:
            data = s[0]
            labels = s[1]
            suffix = s[2]
            features = model.predict(data)
            data_filename = constants.data_filename(data_prefix + suffix, es, fold)
            features_filename = constants.data_filename(
                features_prefix + suffix, es, fold
            )
            labels_filename = constants.data_filename(labels_prefix + suffix, es, fold)
            np.save(data_filename, data)
            np.save(features_filename, features)
            np.save(labels_filename, labels)


class DataGenerator(Sequence):
    def __init__(self, data, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_data, {'decoder': batch_data}


class DataGeneratorForTraining(Sequence):
    def __init__(self, data, labels, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.data, self.labels = data, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_data, {'classifier': batch_labels, 'decoder': batch_data}
