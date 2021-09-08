---
layout: default
title: Model
nav_order: 1
parent: "mesonet Package"
grand_parent: "Part 1: Model Construction and Training"
---

| **Module** | `model` |
| **Source** | <https://bit.ly/3v75NGQ> |
| **Description** | Builds and obtains predictions from MesoNet variants |
| **Import** | `import mesonet.model` |
| **Depends on** | `mesonet.data` |

## <!-- omit in toc --> Jump To

- [Core Functions](#core-functions)
  - [`activation_layer(ip, activation, *args)`](#activation_layerip-activation-args)
  - [`conv2D(ip, filters, kernel_size, activation, padding="same", pool_size=(2, 2))`](#conv2dip-filters-kernel_size-activation-paddingsame-pool_size2-2)
  - [`fully_connected_layer(ip, units, activation, dropout)`](#fully_connected_layerip-units-activation-dropout)
  - [`build_model(ip=Input(shape=(IMG_WIDTH, IMG_WIDTH, 3)), activation="relu", dropout=0.5, hidden_activation="lrelu")`](#build_modelipinputshapeimg_width-img_width-3-activationrelu-dropout05-hidden_activationlrelu)
  - [`predict(model, data, steps=None, threshold=0.5)`](#predictmodel-data-stepsnone-threshold05)
  - [`make_prediction(model_path, data_dir, threshold=0.5, batch_size=64, return_probs=False, return_report=False)`](#make_predictionmodel_path-data_dir-threshold05-batch_size64-return_probsfalse-return_reportfalse)
- [Helper Functions](#helper-functions)
  - [`get_loaded_model(path)`](#get_loaded_modelpath)
  - [`get_activation_model(model, conv_idx)`](#get_activation_modelmodel-conv_idx)
  - [`evaluate_model(model, test_data_dir, batch_size=64)`](#evaluate_modelmodel-test_data_dir-batch_size64)
  - [`get_classification_report(true, preds, output_dict=False)`](#get_classification_reporttrue-preds-output_dictfalse)
  - [`save_model_history(history, filename)`](#save_model_historyhistory-filename)

## Core Functions

### `activation_layer(ip, activation, *args)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/a39ffff11bfb2512cb5fca137bd29b9c47d2d54b/mesonet/model.py#L23)

Function to obtain an activation layer with the given input.

It initializes a ReLU, ELU or LeakyReLU activation layer with the given input layer based on `activation`. This function can be used in place of the `activation` keyword argument in all Keras layers to mix-match activations for different layers and easily use ELU and LeakyReLU, which otherwise need to be imported separately.

| **Arguments** |                                                                                                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ip`          | `tf.keras.layers`, Numpy array or `list`-like: Input for the layer.                                                                                                               |
| `activation`  | `str`: Required activation layer. It can be:<br> - relu: Returns `tf.keras.layers.ReLU`<br> - elu: Returns `tf.keras.layers.ELU`<br> - lrelu: Returns `tf.keras.layers.LeakyReLU` |
| `*args`       | `list`-like: Any additional arguments to be passed when instantiating the layer.                                                                                                  |
| **Returns**   | A `tf.keras.layers` instance initialized with the given arguments and passed the given input.                                                                                     |
| **Raises**    | `KeyError` when `activation` is not one of the specified values.                                                                                                                  |

#### <!-- omit in toc --> Example Usage

Here, we create a convolutional layer with 8 kernels of size 3 x 3 each, which takes 256 x 256 x 3 images as its input and uses a ReLU activation.

```python
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from mesonet.model import activation_layer

# Create input layer
ip = Input(shape=(256, 256, 3))
# Create convolutional layer
layer = Conv2D(filters=8, kernel_size=(3, 3))(ip)
# Add ReLU activation
layer = activation_layer(layer, 'relu')
```

### `conv2D(ip, filters, kernel_size, activation, padding="same", pool_size=(2, 2))`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/a39ffff11bfb2512cb5fca137bd29b9c47d2d54b/mesonet/model.py#L48)

Function to obtain a convolutional "block". A convolutional block is defined as a set of layers where the first layer is a convolutional layer. The entire set of layers is (in this order):

- `tf.keras.layers.Conv2D` - Convolutional layer.
- `tf.keras.layers.ReLU`, `tf.keras.layers.ELU` or `tf.keras.layers.LeakyReLU` - Activation layer.
- `tf.keras.layers.BatchNormalization` - Batch normalization layer.
- `tf.keras.layers.MaxPooling2D` - Max pooling layer.

It feeds the given input to the convolutional layer and then successively feeds the outputs of one layer to the layer below it, thus encapsulating the entire convolutional operation applied in the original MesoNet paper and providing a reusable way to replicate the same operation multiple times.

| **Arguments** |                                                                                                                                                                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ip`          | `tf.keras.layers`, Numpy array or `list`-like: Input for the Conv2D layer.                                                                                                                                                                      |
| `filters`     | `int`: Number of filters in the Conv2D layer.                                                                                                                                                                                                   |
| `kernel_size` | `int` or `list`-like with 2 integers: Size of each filter in the Conv2D layer. Specifies the height and width of the filters. When an int, the height and width are the same.                                                                   |
| `activation`  | `str`: Required activation layer. It can be:<br> - relu: Returns `tf.keras.layers.ReLU`<br> - elu: Returns `tf.keras.layers.ELU`<br> - lrelu: Returns `tf.keras.layers.LeakyReLU`                                                               |
| `padding`     | `str`: One of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. Defaults to "same". |
| `pool_size`   | `int` or tuple of 2 integers: Size of pooling window for the pooling layer. Specifies the height and width of the window. When an int, the height and width are the same. Defaults to (2, 2).                                                   |
| **Returns**   | A `tf.keras.layers` instance encapsulating the block.                                                                                                                                                                                           |

> **Note**: The stride value for the Conv2D layer is always the default value of `1`.

#### <!-- omit in toc --> Example Usage

Here, we build a complete TensorFlow model with 3 convolutional blocks, each with 16 filters of size 3 x 3 and ReLU activation, and one fully-connected layer with 16 neurons.

```python
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense
from mesonet.model import conv2D

# Create input layer
ip = Input(shape=(256, 256, 3))
layer = ip

# Add the 3 convolutional layers with a simple loop
for i in range(3):
    layer = conv2D(layer, filters=16, kernel_size=3, 'relu')

# Add the fully-connected layer
layer = Dense(16)(layer)
# Add an output layer
op_layer = Dense(1, activation='sigmoid')(layer)

# Create model
model = Model(ip, op_layer)
```

### `fully_connected_layer(ip, units, activation, dropout)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/a39ffff11bfb2512cb5fca137bd29b9c47d2d54b/mesonet/model.py#L86)

Function to obtain an fully-connected "block". A fully-connected block is defined as a set of layers where the first layer is a fully-connected (dense) layer. The entire set of layers is (in this order):

- `tf.keras.layers.Dense` - Fully-connected layer.
- `tf.keras.layers.ReLU`, `tf.keras.layers.ELU` or `tf.keras.layers.LeakyReLU` - Activation layer.
- `tf.keras.layers.Dropout` - Dropout layer.

It feeds the given input to the dense layer and then successively feeds the outputs of one layer to the layer below it, thus encapsulating the entire hidden layer operation applied in the original MesoNet paper and providing a reusable way to replicate the same operation multiple times.

| **Arguments** |                                                                                                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ip`          | `tf.keras.layers`, Numpy array or `list`-like: Input for the Dense layer.                                                                                                                    |
| `units`       | `int`: Number of neurons in the Dense layer.                                                                                                                                                 |
| `activation`  | `str`: Required activation layer. It can be:<br> - relu: ReLU activation.<br> - elu: ELU activation.<br> - lrelu: LeakyReLU activation.<br>The alpha value for the activation is always 0.1. |
| `dropout`     | `float`: Rate of dropout (between 0 and 1) for the Dropout layer.                                                                                                                            |
| **Returns**   | A `tf.keras.layers` instance encapsulating the block.                                                                                                                                        |

#### <!-- omit in toc --> Example Usage

Here, we build a simple fully-connected network with a 1024-neuron input layer, 3 ReLU-activated hidden layers with 64, 16 and 8 neurons respectively, and an output layer with a sigmoid activation. A dropout of 0.5 is applied to each hidden layer.

```python
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from mesonet.model import fully_connected_layer

ip = Input(shape=(1, 1024))

neurons = [64, 16, 8]
layer = ip
for x in neurons:
    layer = fully_connected_layer(layer, x, 'relu', 0.5)

op_layer = Dense(1, activation='sigmoid')(layer)

model = Model(ip, op_layer)
```

### `build_model(ip=Input(shape=(IMG_WIDTH, IMG_WIDTH, 3)), activation="relu", dropout=0.5, hidden_activation="lrelu")`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/605041c2316ae6017e0291082952fd062ef72a39/mesonet/model.py#L111)

Function to obtain a model exactly replicating the architecture in the paper.

See the model schematic [here](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/imgs/model_schematic.png).

It acts as a shortcut function to directly obtain a model with the same architecture as the paper and start training it, while also allowing you to modify the shape of the input, the activations and the dropout. Use this if you don't want to create your own variant with a different architecture.

| **Arguments**       |                                                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ip`                | `tf.keras.Input`: Represents the input characteristics of the model. It should match the dimensions of the images in the dataset. Defaults to Input(shape=(IMG_WIDTH, IMG_WIDTH, 3)). |
| `activation`        | `str`: Activation function for the convolutional layers. Can be:<br> - relu: ReLU activation.<br> - elu: ELU activation.<br> - lrelu: LeakyReLU activation.<br>Defaults to "relu".    |
| `dropout`           | `float`: Rate of dropout (between 0 and 1) for the Dropout layer.                                                                                                                     |
| `hidden_activation` | `str`: Activation function after the hidden Dense layer. Can be: <br> - relu: ReLU activation.<br> - elu: ELU activation.<br> - lrelu: LeakyReLU activation.<br>Defaults to "lrelu".  |
| **Returns**         | A `tf.keras.Model` instance encapsulating the built model                                                                                                                             |

#### <!-- omit in toc --> Example Usage

When called with no arguments, the function gives you the exact model used in the paper. To verify this, you can call the `.summary()` method on the returned `tf.keras.Model` instance.

```python
from mesonet.model import build_model

meso = build_model()
meso.summary()
```

### `predict(model, data, steps=None, threshold=0.5)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/605041c2316ae6017e0291082952fd062ef72a39/mesonet/model.py#L220)

Function to make predictions on data. Can only handle a model with a sigmoid activation at the output layer.

It calls the `.predict()` method on the given model to obtain the predicted sigmoid probabilities for the data, and then classifies them into the positive (`1`) or the negative class (`0`) based on the threshold. The function is agnostic to the actual class labels.

| **Arguments** |                                                                                                                                                                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `tf.keras.Model`: Model to use for making predictions.                                                                                                                                                                                                |
| `data`        | Data on which to make predictions. See <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>                                                                                                                                            |
| `steps`       | `int`: Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None. If data is a tf.data dataset and steps is None, predict will run until the input dataset is exhausted.      |
| `threshold`   | `float`: Minimum probability (between 0 and 1) to classify a prediction as the positive class (`1`). Comparison used is `>=`. Defaults to 0.5.                                                                                                        |
| **Returns**   | A two-`tuple` (x, y) where:<br> 1. x is a Numpy array of dimension (NUM_IMGS, 1) containing the predicted sigmoid probabilities for each image.<br> 2. y is a Numpy array of dimension (NUM_IMGS, 1) containing the predicted classes for each image. |

### `make_prediction(model_path, data_dir, threshold=0.5, batch_size=64, return_probs=False, return_report=False)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/605041c2316ae6017e0291082952fd062ef72a39/mesonet/model.py#L286)

Function to obtain predictions from a saved model using data from a directory.

It uses the `predict()` function to make predictions using the given model on the data loaded from a directory. Instead of returning `1` or `0` as the classes, it replaces `1` and `0` with the corresponding label (e.g. `real` and `deepfake`). Optionally, it can return the predicted probabilities and also generates a ROC report (using `sklearn.metrics.classification_report`).

It is useful when you have more user-facing tasks such as displaying the results in a window. In fact, the [prediction sub-command](../cli/predict_command.md) of the CLI uses this function to do its work.

| **Arguments**   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_path`    | `str`: Path to the saved model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `data_dir`      | `str`: Path to the data on which predictions are to be made.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `threshold`     | `float`: Minimum probability (between 0 and 1) to classify a prediction as the positive class (`1`). Comparison used is `>=`. The function also replaces `0`/`1` with the class labels. Defaults to 0.5.                                                                                                                                                                                                                                                                                              |
| `batch_size`    | `int`: Size of batches of data, used in initializing the data generator. Defaults to 64.                                                                                                                                                                                                                                                                                                                                                                                                              |
| `return_probs`  | `bool`: If True, along with predicted labels, the predicted probabilities are also included in the result. The probabilities are converted to %. Since each predicted probability represents the probability of the sample being in the positive class, for samples belonging to the negative class, the probability is subtracted from 1. Default to False.                                                                                                                                          |
| `return_report` | `bool`: If True, an ROC report is also included in the result a str. Defaults to False.                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Returns**     | A two-`tuple` (x, y) where:<br>1. x is a Numpy array of dimension (NUM_IMGS, 2) when `return_probs` is False and of (NUM_IMGS, 3) when `return_probs` is True. For each image, a row vector in the format `[filename, label, <prob>]` is created, where `filename` refers to the name of the image file, `label` is the predicted label and `<prob>` is the optional predicted probability.<br>2. y is an empty string when `return_report` is False and the ROC report when `return_report` is True. |

#### <!-- omit in toc --> Example Usage

Here, we make predictions on data stored in the directory `data/test` using a model saved as `model.hdf5`. We also want a classification report and so, set `return_report` to `True`.

```python
from mesonet.model import make_prediction

preds, report = make_prediction('model.hdf5', 'data/test', return_report=True)
```

## Helper Functions

### `get_loaded_model(path)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/605041c2316ae6017e0291082952fd062ef72a39/mesonet/model.py#L159)

Function to load a saved model from a H5 file or .pd folder.

It takes the given path and uses `tf.keras.models.load_model` to load the saved model at the path as a `tf.keras.Model` instance.

| **Arguments** |                                                 |
| ------------- | ----------------------------------------------- |
| `path`        | `str`: Path to the saved model.                 |
| **Returns**   | A `tf.keras.Model` instance of the saved model. |

> Note: The returned model is NOT compiled.

#### <!-- omit in toc --> Example Usage

If a model is saved in a file called `model.hdf5`, it can be loaded as:

```python
from mesonet.model import get_loaded_model

model = get_loaded_model('model.hdf5')
```

### `get_activation_model(model, conv_idx)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/605041c2316ae6017e0291082952fd062ef72a39/mesonet/model.py#L172)

Function to obtain the "activation model" of a trained model.

An activation model is defined as a model which has the same input as the trained model and whose output consists of the output of least one
convolutional layer of the trained model. Since this outputs are obtained after an activation function is applied on the convolution, hence the name.

It automatically detects all the convolutional layers present in the trained model, selects the specified ones from among them and creates a new `tf.keras.Model` instance which has the same input(s) as the trained model and the outputs of the selected convolutional layers.

When you make predictions using this model, the result will be a list with as many Numpy arrays as the number of convolutional layers selected, each array corresponding to the output of a layer.

It is useful for visualizing the convolutional layers as images.

| **Arguments** |                                                                                                                                                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `tf.keras.Model`: Trained model whose activation model is required.                                                                                                                                                               |
| `conv_idx`    | `list`-like: Indices of the convolutional layers which should be included in the output of the activation model (0-indexed). The ordering of indices is important since the outputs (after prediction) will be in the same order. |
| **Returns**   | A `tf.keras.Model` instance representing the activation model.                                                                                                                                                                    |

> Note: If you create an activation model with a single layer, you get a single Numpy array containing the output of that layer (not a list) when making predictions.

#### <!-- omit in toc --> Example Usage

The following obtains a model using `build_model`, trains it and then obtains an activation model with the first and last convolutional layers as the outputs. Note that since there are four convolutional layers in the original MesoNet architecture, the index of the last convolutional layer is 3.

```python
from mesonet.model import build_model, get_activation_model

model = build_model()
# Code to train the model
...
# Create activation model
activation_model = get_activation_model(model, [0, 3])
```

### `evaluate_model(model, test_data_dir, batch_size=64)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/605041c2316ae6017e0291082952fd062ef72a39/mesonet/model.py#L202)

Function to obtain evaluation metrics on a trained model.

It loads the data from the given directory and uses the `.evaluate()` method to obtain the evaluation metrics (as used when training the model) of the given trained model.

For information on the expected directory structure, check the [documentation](./data.md) for the `data` module.

| **Arguments**   |                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------- |
| `model`         | `tf.keras.Model`: Trained model whose metrics are required.                                  |
| `test_data_dir` | `str`: Path to directory containing the data to be used for evaluation.                      |
| `batch_size`    | `int`: Size of batches of the data. Defaults to 64. You'll probably not need to change this. |
| **Returns**     | A `list` with values for all the metrics used when the model was trained.                    |

#### <!-- omit in toc --> Example Usage

Here, we load a model saved at `model.hdf5` using `get_loaded_model()` and then use `evaluate_model()` on data located in `data/test` to obtain the evaluation metrics.

```python
from mesonet.module import get_loaded_model, evaluate_model

model = get_loaded_model('model.hdf5')
metrics = evaluate_model(model, 'data/test')
```

### `get_classification_report(true, preds, output_dict=False)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/605041c2316ae6017e0291082952fd062ef72a39/mesonet/model.py#L251)

Function to obtain an ROC report for a set of predictions.

It uses `sklearn.metrics.classification_report()` to obtain the recall, precision and F1-scores, returning them as a string or a dictionary.

| **Arguments** |                                                                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `true`        | 1D array-like: True labels.                                                                                                                 |
| `preds`       | 1D array-like: Predicted labels.                                                                                                            |
| `output_dict` | `bool`: If True, outputs a dict rather than a str. Defaults to False.                                                                       |
| **Returns**   | 1. When `output_dict` is False, a nicely formatted string with the ROC report.<br>2. When `output_dict` is True, a dict with the ROC report |

### `save_model_history(history, filename)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/605041c2316ae6017e0291082952fd062ef72a39/mesonet/model.py#L348)

Function to dump a model's history into a pickle file.

This is useful in scenarios where you want to generate plots even after you have terminated the current session.

| **Arguments** |                                                                         |
| ------------- | ----------------------------------------------------------------------- |
| `history`     | `History` instance: History of the model.                               |
| `filename`    | `str`: Filename of the target file to which the history will be dumped. |

{: .text-center}
[Back to top](#-jump-to)
