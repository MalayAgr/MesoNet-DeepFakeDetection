import pickle

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    ELU,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
    MaxPooling2D,
    ReLU,
)
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


def fully_connected_layer(ip, units, activation, dropout):
    """
    Function to obtain an FCC 'block' with the following layers:
        - Dense layer with 16 neurons.
        - Activation
        - Dropout

    Args:
        ip (tf.keras.layers, Numpy array or list-like): Input for the Dense layer.
        units (int): Number of neurons in the Dense layer.
        activation (str): Activation function to use. Can be:
            - 'relu': ReLU activation.
            - 'elu': ELU activation.
            - 'lrelu': LeakyReLU activation.
            The alpha value for the activation is always 0.1.
        dropout (float): Rate of dropout (between 0 and 1) for the Dropout layer.

    Returns:
        A tf.keras.layers instance encapsulating the block.
    """
    layer = Dense(units)(ip)
    layer = activation_layer(layer, activation, *[0.1])
    return Dropout(rate=dropout)(layer)


def build_model(
    ip=Input(shape=(IMG_WIDTH, IMG_WIDTH, 3)),
    activation="relu",
    dropout=0.5,
    hidden_activation="lrelu",
):
    """
    Function to obtain a model exactly replicating the architecture in the paper.
    See: https://arxiv.org/pdf/1809.00888v1.pdf and imgs/model_schematic.png.
    Note that the model is NOT compiled.

    Args:
        ip (tf.keras.Input): Represents the input characteristics of the model.
            It should match the dimensions of the images in the dataset.
            Defaults to Input(shape=(IMG_WIDTH, IMG_WIDTH, 3)).
        activation (str): Activation function for the conv layers. Defaults to "relu".
        dropout (float): Rate of dropout (between 0 and 1) for the Dropout layer.
            Defaults to 0.5.
        hidden_activation (str): Activation function after the hidden Dense layer.
            Defaults to "lrelu".

    Returns:
        A tf.keras.Model instance encapsulating the built model.
    """
    layer = conv2D(ip, filters=8, kernel_size=(3, 3), activation=activation)

    layer = conv2D(layer, filters=8, kernel_size=(5, 5), activation=activation)

    layer = conv2D(layer, filters=16, kernel_size=(5, 5), activation=activation)

    layer = conv2D(
        layer, filters=16, kernel_size=(5, 5), activation=activation, pool_size=(4, 4)
    )

    layer = Flatten()(layer)
    layer = Dropout(rate=dropout)(layer)

    layer = fully_connected_layer(
        layer, units=16, activation=hidden_activation, dropout=dropout
    )

    op_layer = Dense(1, activation="sigmoid")(layer)

    model = Model(ip, op_layer)

    return model


def get_loaded_model(path):
    """
    Function to load a saved model from a H5 file or .pd folder.

    Args:
        path (str): Path to the saved model.

    Returns:
        A tf.keras.Model instance with the loaded model.
    """
    return load_model(path)


def get_activation_model(model, conv_idx):
    """
    Function to obtain the "activation model" of a trained model.
    An activation model is defined as a model which has the same input
    As the trained model and whose output consists of the output at least one
    Convolutional layer for the trained model. Since this outputs are obtained
    After an activation function, hence the name.

    Args:
        model (tf.keras.Model): Model whose activation model is required.
        conv_idx (list-like): Indices of the conv layers which should
            be included in the output of the activation model (0-indexed).
            The ordering of indices is important since the outputs (after prediction)
            will be in the same order.

    Returns:
        A tf.keras.Model instance representing the activation model.
    """

    if not conv_idx:
        raise ValueError("conv_idx requires at least one element")

    conv_layers = [layer for layer in model.layers if "conv" in layer.name]
    selected_layers = [conv_layers[i] for i in conv_idx]
    activation_model = Model(
        inputs=model.inputs, outputs=[layer.output for layer in selected_layers]
    )
    return activation_model


def evaluate_model(model, test_data_dir, batch_size=64):
    """
    Function to obtain evaluation metrics on a trained model.

    Args:
        model (tf.keras.Model): Model whose metrics are required.
        test_data_dir (str): Path to directory containing the data to be used
            for evaluation.
        batch_size (int): Size of batches of the data. Defaults to 64.

    Returns:
        A list with values for all the metrics used when the model
        was originally trained.
    """
    data = get_test_data_generator(test_data_dir, batch_size)
    return model.evalute(data)


def predict(model, data, steps=None, threshold=0.5):
    """
    Function to make predictions on data.
    Note that this function can only handle a model with a sigmoid activation
    At the output layer.

    Args:
        model (tf.keras.Model): Model to use for making predictions.
        data: See https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict.
        steps (int): Total number of steps (batches of samples) before declaring the
            prediction round finished. Ignored with the default value of None.
            If data is a tf.data dataset and steps is None, predict will run until
            the input dataset is exhausted.
        threshold (float): Minimum probability to classify a prediction as
            the positive class ('1'). Comparison used is >=. Defaults to 0.5.

    Returns:
        A two-tuple (x, y) where:
            1. x is a Numpy array of dimension (NUM_IMGS, 1) containing the predicted
                sigmoid probabilities for each image.
            2. y is a Numpy array of dimension (NUM_IMGS, 1) containing the
                predicted classes for each image.
    """
    probs = model.predict(data, steps=steps, verbose=1)
    preds = np.where(probs >= threshold, 1, 0)

    probs, preds = probs.reshape(-1), preds.reshape(-1)

    return probs, preds


def get_classification_report(true, preds, output_dict=False):
    """
    Function to obtain an ROC report for a set of predictions.

    Args:
        true (1D array-like): True labels.
        preds (1D array-like): Predicted labels.
        output_dict (bool): If True, outputs a dict rather than a str.
            Defaults to False.

    Returns:
        1. When output_dict is False, a nicely formatted string with the ROC report.
            Example:
                        precision    recall  f1-score   support

                0       0.96      0.94      0.95       773
                1       0.96      0.97      0.97      1172

         accuracy                           0.96      1945
        macro avg       0.96      0.96      0.96      1945
     weighted avg       0.96      0.96      0.96      1945

        2. When output_dict is True, a dict with the ROC report:
            Example:
                {'label 1': {'precision':0.5,
                             'recall':1.0,
                             'f1-score':0.67,
                             'support':1},
                 'label 2': { ... },
                 ...
                 }
    """
    return classification_report(true, preds, output_dict=output_dict)


def make_prediction(
    model_path,
    data_dir,
    threshold=0.5,
    batch_size=64,
    return_probs=False,
    return_report=False,
):
    """
    Function to obtain predictions from a saved model using data from a directory.

    Args:
        model_path (str): Path to the saved model.
        data_dir (str): Path to the data on which predictions are to be made.
        threshold (float): Minimum probability to classify a prediction as
            the positive class ('1'). Comparison used is >=. The function also replaces
            0/1 with the actual class names. Defaults to 0.5.
        batch_size (int): Size of batches of data, used in initializing the data generator.
            Defaults to 64.
        return_probs (bool): If True, along with predicted labels, the predicted
            probabilities are also included in the result. The probabilities are
            converted to %. Since each predicted probability represents the
            probability of the sample being in the +ve class, for samples belonging
            to the -ve class, the probability is subtracted from 1.
        return_report (bool): If True, an ROC report is also included in the result
            as a str. Defaults to False.

    Returns:
        A two-tuple (x, y) where:
            1. x is a Numpy array of dimension (NUM_IMGS, 2) when return_probs is
                False and of (NUM_IMGS, 3) when return_probs is True. For each image,
                a row vector in the format [filename, label, <prob>] is created,
                where filename refers to the name of the image file, label is the
                predicted label and <prob> is the optional predicted probability.
            2. y is an empty string when return_report is False and the ROC report
                when return_report is True.
    """
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
    """
    Function to dump a model's history into a pickle file.
    This is useful in scenarios where you want to generate plots
    Even after you have terminated the current sessions.

    Args:
        history (History instance): History of the model.
        filename (str): Filename of the target file to which the history will be dumped.
    """
    with open(filename, "wb") as f:
        pickle.dump(history.history, f)
