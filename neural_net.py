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

import math
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Dropout,
    Dense,
    LeakyReLU,
    Flatten,
    Reshape,
    UpSampling2D,
    BatchNormalization,
    LayerNormalization,
    SpatialDropout2D,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import constants
import dataset

epochs = 300
patience = 10
truly_training_percentage = 0.80


def conv_block(entry, layers, filters, dropout, first_block=False, pooling=True):
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
    output = (
        entry
        if not pooling
        else MaxPool2D(pool_size=2, strides=2, padding='same')(entry)
    )
    output = SpatialDropout2D(dropout)(output)
    return output


# The number of layers defined in get_encoder.
encoder_nlayers = 40


def get_encoder(domain):
    dropout = 0.1
    input_data = Input(shape=(dataset.rows, dataset.columns, 1))
    filters = domain // 16
    output = conv_block(input_data, 2, filters, dropout, first_block=True)
    filters *= 2
    dropout += 0.025
    output = conv_block(output, 2, filters, dropout)
    filters *= 2
    dropout += 0.025
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout += 0.025
    output = conv_block(output, 3, filters, dropout)
    filters *= 2
    dropout += 0.025
    output = conv_block(output, 3, filters, dropout)

    # --- THE FEATURE BOOSTER ---
    # We add a 2*domain-filter block here to capture fine-grained textures.
    # But we DO NOT increase the final domain size.
    # dropout *= 2.0
    # output = Conv2D(2 * domain, kernel_size=3, padding='same', activation='relu')(
    #     output
    # )
    # output = BatchNormalization()(output)
    # output = SpatialDropout2D(0.2)(output)
    # --------------------------------------

    output = Flatten()(output)  
    # output = Dense(constants.domain, name='domain_layer')(output)
    # output = LayerNormalization()(output)
    return input_data, output


def get_decoder(domain):
    n = int(math.log2(domain))
    remainer = 3 if (n % 2 != 0) else 2
    initial_divisor = 2 * remainer
    iter_divisor = 2 ** ((n - remainer) // 2 - 1)

    input_mem = Input(shape=(domain,))
    # Which is going to be multiplied by two by each Conv2DTranspose layer in the loop.
    width = dataset.columns // 4
    filters = domain // initial_divisor
    dense = Dense(width * width * filters, activation='relu')(input_mem)
    output = Reshape((width, width, filters))(dense)
    dropout = 0.1
    for i in range(2):
        filters = filters // iter_divisor
        output = UpSampling2D(size=(2, 2))(output)
        output = Conv2D(filters, (3, 3), padding='same')(output)
        output = BatchNormalization()(output)  # Optional in decoder
        output = LeakyReLU(alpha=0.2)(output)
        output = SpatialDropout2D(dropout)(output)
        dropout /= 2.0
    output = Conv2D(
        filters=1, kernel_size=3, strides=1, activation='sigmoid', padding='same'
    )(output)
    return input_mem, output


# The number of layers defined in get_classifier.
classifier_nlayers = 6


def get_classifier(domain):
    input_mem = Input(shape=(domain,))
    # Uses LeakyReLU or ELU, as they allow negative values to pass through,
    # so the classifier can "see" the full latent space.
    dense = Dense(4 * domain)(input_mem)
    dense = LeakyReLU(negative_slope=0.1)(dense)
    drop = Dropout(0.2)(dense)
    dense = Dense(2 * domain)(drop)
    dense = LeakyReLU(negative_slope=0.1)(dense)
    drop = Dropout(0.2)(dense)
    # dense = Dense(domain)(drop)
    # dense = LeakyReLU(negative_slope=0.1)(dense)
    # drop = Dropout(0.2)(dense)
    # dense = Dense(domain // 2)(drop)
    # dense = LeakyReLU(negative_slope=0.1)(dense)
    drop = Dropout(0.2)(dense)
    classification = Dense(
        constants.network_labels, activation='softmax', name='classified'
    )(drop)
    return input_mem, classification


def train_network(prefix):
    confusion_matrix = np.zeros((constants.network_labels, constants.network_labels))
    histories = []
    strategy = tf.distribute.MirroredStrategy()
    for fold in range(constants.n_folds):
        print(f'FOLD: {fold}')
        print('Getting the dataset ready...')
        training_gen = dataset.get_training(fold, categorical=True)
        # No shuffling is needed for validation nor testing.
        validating_gen = dataset.get_validating(fold, categorical=True)
        testing_gen = dataset.get_testing(fold, categorical=True)
        predict_gen = dataset.get_testing(fold, predict_only=True)

        rmse = tf.keras.metrics.RootMeanSquaredError()
        with strategy.scope():
            domain = constants.domain
            print('Building and compiling the neural network...')
            input_data = Input(shape=(dataset.rows, dataset.columns, 1))
            input_enc, output_enc = get_encoder(domain)
            input_class, output_class = get_classifier(domain)
            input_dec, output_dec = get_decoder(domain)

            encoder = Model(input_enc, output_enc, name='encoder')
            encoder.summary()
            classifier = Model(input_class, output_class, name='classifier')
            classifier.summary()
            decoder = Model(input_dec, output_dec, name='decoder')
            decoder.summary()
            encoded = encoder(input_data)
            decoded = decoder(encoded)
            classified = classifier(encoded)

            decoder_weight_var = tf.Variable(0.0, dtype=tf.float32, trainable=False)
            warmup_cb = DecoderWeightScheduler(decoder_weight_var, linear_warmup)

            model = Model(
                inputs=input_data,
                outputs={'classifier': classified, 'decoder': decoded},
            )
            model.compile(
                loss=['categorical_crossentropy', 'mean_squared_error'],
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=1e-3
                ),  # Learning rate for a batch size of 2048
                loss_weights={'classifier': 1, 'decoder': decoder_weight_var},
                metrics={'classifier': 'accuracy', 'decoder': rmse},
            )
            model.summary()

            full_classifier = Model(
                inputs=input_data, outputs=classified, name='full_classifier'
            )
            # autoencoder = Model(inputs=input_data, outputs=decoded, name='autoencoder')

        print('Training the neural network...')
        early_stopping = EarlyStopping(
            monitor='val_classifier_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=2,
        )

        lr_reducer = ReduceLROnPlateau(
            monitor='val_classifier_accuracy',
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            mode='max',
            verbose=2,
        )

        history = model.fit(
            training_gen,
            # batch_size=constants.batch_size,
            epochs=epochs,
            validation_data=validating_gen,
            callbacks=[early_stopping, lr_reducer, warmup_cb],
            verbose=2,
        )
        histories.append(history)
        history = model.evaluate(testing_gen, return_dict=True)
        histories.append(history)
        print('Creating the confusion matrix...')
        predicted_labels = np.argmax(full_classifier.predict(predict_gen), axis=1)
        # Retrieve True Labels directly from HDF5 using generator indices
        true_labels = predict_gen.get_all_labels()
        confusion_matrix += tf.math.confusion_matrix(
            true_labels,
            predicted_labels,
            num_classes=constants.network_labels,
        )
        print('Saving everything needed for the future...')
        encoder.save(constants.encoder_filename(prefix, fold))
        decoder.save(constants.decoder_filename(prefix, fold))
        classifier.save(constants.classifier_filename(prefix, fold))
        name = constants.classification_name()
        prediction_filename = constants.data_filename(name, es=None, fold=fold)
        np.save(prediction_filename, predicted_labels)
    history_record = {
        'metadata': {
            'batch_size': constants.batch_size,
            'epochs': epochs,
            'n_folds': constants.n_folds,
        },
        'results': histories,
    }
    confusion_matrix = confusion_matrix.numpy()
    totals = confusion_matrix.sum(axis=1).reshape(-1, 1)
    return history_record, confusion_matrix / totals


def obtain_features(model_prefix, features_prefix, labels_prefix):
    for fold in range(constants.n_folds):
        # Load the encoder
        filename = constants.encoder_filename(model_prefix, fold)
        model = tf.keras.models.load_model(filename)

        # 1. Get Generators (which replace the raw data arrays)
        # We set predict_only=True so the generator returns ONLY images for model.predict.
        fill_gen = dataset.get_filling(fold, predict_only=True)
        test_gen = dataset.get_testing(fold, predict_only=True)
        settings = [
            (fill_gen, constants.filling_suffix),
            (test_gen, constants.testing_suffix),
        ]

        for gen, suffix in settings:
            print(f'Generating features for {suffix}...')
            features = model.predict(
                gen,
                verbose=1,
            )
            labels = gen.get_all_labels()
            features_filename = constants.shared_data_filename(
                features_prefix + suffix, fold
            )
            labels_filename = constants.shared_data_filename(
                labels_prefix + suffix, fold
            )

            print('Saving features and labels ...')
            np.save(features_filename, features)
            np.save(labels_filename, labels)


class DecoderWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, weight_var, schedule_fn):
        super(DecoderWeightScheduler, self).__init__()
        self.weight_var = weight_var
        self.schedule_fn = schedule_fn

    def on_epoch_begin(self, epoch, logs=None):
        new_weight = self.schedule_fn(epoch)
        # Use assign to update the tensor value without re-compiling
        self.weight_var.assign(new_weight)
        print(f'\n - current_decoder_weight: {self.weight_var.numpy():.4f}')


def linear_warmup(epoch):
    start_weight = 0.0
    end_weight = 10.0
    warmup_epochs = 160  # Reach full weight by epoch, then keep it constant

    if epoch < warmup_epochs:
        return start_weight + (end_weight - start_weight) * (epoch / warmup_epochs)
    return end_weight
