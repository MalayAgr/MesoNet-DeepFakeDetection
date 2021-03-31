import pickle

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (ELU, BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, LeakyReLU, MaxPooling2D,
                                     ReLU)

from utils import IMG_WIDTH


def activation_layer(ip, activation, *args):
    return {
        'relu': ReLU(*args)(ip),
        'elu': ELU(*args)(ip),
        'lrelu': LeakyReLU(*args)(ip)
    }[activation]


def conv2D(
    ip, filters, kernel_size,
    activation, padding='same', pool_size=(2, 2)
):
    layer = Conv2D(
        filters, kernel_size=kernel_size, padding=padding
    )(ip)

    layer = activation_layer(layer, activation=activation)

    layer = BatchNormalization()(layer)

    return MaxPooling2D(pool_size=pool_size, padding=padding)(layer)


def fully_connected_layer(
    ip, hidden_activation, dropout
):
    layer = Dense(16)(ip)
    layer = activation_layer(layer, hidden_activation, *[0.1])
    return Dropout(rate=dropout)(layer)


def build_model(
    ip=Input(shape=(IMG_WIDTH, IMG_WIDTH, 3)),
    activation='relu',
    dropout=0.5,
    hidden_activation='lrelu'
):
    layer = conv2D(
        ip, filters=8, kernel_size=(3, 3), activation=activation
    )

    layer = conv2D(
        layer, filters=8, kernel_size=(5, 5), activation=activation
    )

    layer = conv2D(
        layer, filters=16, kernel_size=(5, 5), activation=activation
    )

    layer = conv2D(
        layer, filters=16, kernel_size=(5, 5),
        activation=activation, pool_size=(4, 4)
    )

    layer = Flatten()(layer)
    layer = Dropout(rate=dropout)(layer)

    layer = fully_connected_layer(
        layer, hidden_activation=hidden_activation, dropout=dropout
    )

    op_layer = Dense(1, activation='sigmoid')(layer)

    model = Model(ip, op_layer)

    return model


def predict(model, data, steps=None, threshold=0.5):
    predictions = model.predict(data, steps=steps, verbose=1)
    return np.where(predictions >= threshold, 1, 0)


def save_model_history(history, filename):
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)
