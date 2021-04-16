import pickle

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (ELU, BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, LeakyReLU, MaxPooling2D,
                                     ReLU)
from tensorflow.keras.models import load_model

from .data import get_test_data_generator
from .utils import IMG_WIDTH


def activation_layer(ip, activation, *args):
    """
    Function to obtain an activation layer with the given input.

    Args:
        ip (tf.keras.layers, Numpy array or list-like): Input for the layer.
        activation (str): Required activation layer. It can be:
            - 'relu': Returns tf.keras.layers.ReLU
            - 'elu: Returns tf.keras.layers.ELU
            - 'lrelu': Returns tf.keras.layers.LeakyReLU
        *args (list-like): Any additional arguments to be passed when
            instantiating the layer.
    Returns:
        A tf.keras.layers instance initialized with the given arguments
        and passed the given input.
    Raises:
        KeyError when an invalid activation is passed.
    """
    return {
        "relu": ReLU(*args)(ip),
        "elu": ELU(*args)(ip),
        "lrelu": LeakyReLU(*args)(ip),
    }[activation]


def conv2D(ip, filters, kernel_size, activation, padding="same", pool_size=(2, 2)):
    """
    Function to obtain a convolutional 'block,' with the following layers:
    (in this order)
        - Conv2D
        - Activation
        - BatchNormalization
        - MaxPooling2D

    Args:
        ip (tf.keras.layers, Numpy array or list-like): Input for the Conv2D layer
        filters (int): Number of filters in the Conv2D layer
        kernel_size (int or list-like with 2 integers): Size of each filter in the Conv2D layer.
            Specifies the height and width of the filters. When an int, the height and width
            are the same.
        activation (str): Activation function to use. Can be:
            - 'relu': ReLU activation.
            - 'elu': ELU activation.
            - 'lrelu': LeakyReLU activation.
        padding (str): One of "valid" or "same" (case-insensitive). "valid" means no padding.
            "same" results in padding evenly to the left/right or up/down of the input such
            that output has the same height/width dimension as the input. Defaults to "same".
        pool_size (int or tuple of 2 integers): Size of pooling window. Specifies the height
            and width of the window. When an int, the height and width are the same.
            Defaults to (2, 2).

    Returns:
        A tf.keras.layers instance encapsulating the block.
    """
    layer = Conv2D(filters, kernel_size=kernel_size, padding=padding)(ip)

    layer = activation_layer(layer, activation=activation)

    layer = BatchNormalization()(layer)

    return MaxPooling2D(pool_size=pool_size, padding=padding)(layer)


def fully_connected_layer(ip, activation, dropout):
    """
    Function to obtain an FCC 'block' with the following layers:
        - Dense layer with 16 neurons.
        - Activation
        - Dropout

    Args:
        ip (tf.keras.layers, Numpy array or list-like): Input for the Dense layer.
        activation (str): Activation function to use. Can be:
            - 'relu': ReLU activation.
            - 'elu': ELU activation.
            - 'lrelu': LeakyReLU activation.
            The alpha value for the activation is always 0.1.
        dropout (float): Rate of dropout (between 0 and 1) for the Dropout layer.

    Returns:
        A tf.keras.layers instance encapsulating the block.
    """
    layer = Dense(16)(ip)
    layer = activation_layer(layer, activation, *[0.1])
    return Dropout(rate=dropout)(layer)


def build_model(
    ip=Input(shape=(IMG_WIDTH, IMG_WIDTH, 3)),
    activation="relu",
    dropout=0.5,
    hidden_activation="lrelu",
):
    layer = conv2D(ip, filters=8, kernel_size=(3, 3), activation=activation)

    layer = conv2D(layer, filters=8, kernel_size=(5, 5), activation=activation)

    layer = conv2D(layer, filters=16, kernel_size=(5, 5), activation=activation)

    layer = conv2D(
        layer, filters=16, kernel_size=(5, 5), activation=activation, pool_size=(4, 4)
    )

    layer = Flatten()(layer)
    layer = Dropout(rate=dropout)(layer)

    layer = fully_connected_layer(layer, activation=hidden_activation, dropout=dropout)

    op_layer = Dense(1, activation="sigmoid")(layer)

    model = Model(ip, op_layer)

    return model


def get_loaded_model(path):
    return load_model(path)


def get_activation_model(model, conv_idx):
    conv_layers = [layer for layer in model.layers if "conv" in layer.name]
    selected_layers = [conv_layers[i] for i in conv_idx]
    activation_model = Model(
        inputs=model.inputs, outputs=[layer.output for layer in selected_layers]
    )
    return activation_model


def evaluate_model(model, test_data_dir, batch_size):
    data = get_test_data_generator(test_data_dir, batch_size)
    return model.evalute(data)


def predict(model, data, steps=None, threshold=0.5):
    probs = model.predict(data, steps=steps, verbose=1)
    preds = np.where(probs >= threshold, 1, 0)

    probs, preds = probs.reshape(-1), preds.reshape(-1)

    return probs, preds


def get_classification_report(true, preds, output_dict=False):
    return classification_report(true, preds, output_dict=output_dict)


def make_prediction(
    model_path,
    data_dir,
    threshold=0.5,
    batch_size=64,
    return_probs=False,
    return_report=False,
):
    model = get_loaded_model(model_path)
    data = get_test_data_generator(data_dir, batch_size)

    probs, preds = predict(model, data, steps=None, threshold=threshold)

    report = get_classification_report(data.classes, preds) if return_report else ""

    label_map = {value: key.title() for key, value in data.class_indices.items()}

    preds = np.where(preds == 0, label_map[0], label_map[1])
    preds = preds.reshape(-1)

    stack = (data.filenames, preds)

    if return_probs:
        probs = np.where(probs >= threshold, probs * 100, (1 - probs) * 100)
        probs = probs.reshape(-1)
        stack += (probs,)

    result = np.dstack(stack)
    result = result.reshape(result.shape[1:])

    return result, report


def save_model_history(history, filename):
    with open(filename, "wb") as f:
        pickle.dump(history.history, f)
