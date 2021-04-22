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

- [`mesonet.model.activation_layer(ip, activation, *args)`](#mesonetmodelactivation_layerip-activation-args)
- [`mesonet.model.conv2D(ip, filters, kernel_size, activation, padding="same", pool_size=(2, 2))`](#mesonetmodelconv2dip-filters-kernel_size-activation-paddingsame-pool_size2-2)
- [`mesonet.model.fully_connected_layer(ip, units, activation, dropout)`](#mesonetmodelfully_connected_layerip-units-activation-dropout)

## <!-- omit in toc --> Core Functions

### `mesonet.model.activation_layer(ip, activation, *args)`

[[source]](https://github.com/MalayAgarwal-Lee/MesoNet-DeepFakeDetection/blob/a39ffff11bfb2512cb5fca137bd29b9c47d2d54b/mesonet/model.py#L23)

Function to obtain an activation layer with the given input.

It initializes a ReLU, ELU or LeakyReLU activation layer with the given input layer based on `activation`. This function can be used in place of the `activation` keyword argument in all Keras layers to mix-match activations for different layers and easily use ELU and LeakyReLU, which otherwise need to be imported separately.

| **Arguments** |                                                                                                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ip`          | `tf.keras.layers`, Numpy array or list-like: Input for the layer.                                                                                                                 |
| `activation`  | `str`: Required activation layer. It can be:<br> - relu: Returns `tf.keras.layers.ReLU`<br> - elu: Returns `tf.keras.layers.ELU`<br> - lrelu: Returns `tf.keras.layers.LeakyReLU` |
| `*args`       | list-like: Any additional arguments to be passed when instantiating the layer.                                                                                                    |
| **Returns**   | A `tf.keras.layers` instance initialized with the given arguments and passed the given input.                                                                                     |
| **Raises**    | `KeyError` when `activation` is not one of the specified values.                                                                                                                  |

#### <!-- omit in toc --> Example usage

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

### `mesonet.model.conv2D(ip, filters, kernel_size, activation, padding="same", pool_size=(2, 2))`

[[source]](https://github.com/MalayAgarwal-Lee/MesoNet-DeepFakeDetection/blob/a39ffff11bfb2512cb5fca137bd29b9c47d2d54b/mesonet/model.py#L48)

Function to obtain a convolutional 'block.' A convolutional block is defined as a set of layers where the first layer is a convolutional layer. The entire set of layers is (in this order):

- `tf.keras.layers.Conv2D` - Convolutional layer.
- `tf.keras.layers.ReLU`, `tf.keras.layers.ELU` or `tf.keras.layers.LeakyReLU` - Activation layer.
- `tf.keras.layers.BatchNormalization` - Batch normalization layer.
- `tf.keras.layers.MaxPooling2D` - Max pooling layer.

It feeds the given input to the convolutional layer and then successively feeds the outputs of one layer to the layer below it, thus encapsulating the entire convolutional operation applied in the original MesoNet paper and providing a reusable way to replicate the same operation multiple times.

| **Arguments** |                                                                                                                                                                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ip`          | `tf.keras.layers`, Numpy array or list-like: Input for the Conv2D layer.                                                                                                                                                                        |
| `filters`     | `int`: Number of filters in the Conv2D layer.                                                                                                                                                                                                   |
| `kernel_size` | `int` or list-like with 2 integers: Size of each filter in the Conv2D layer. Specifies the height and width of the filters. When an int, the height and width are the same.                                                                     |
| `activation`  | `str`: Required activation layer. It can be:<br> - relu: Returns `tf.keras.layers.ReLU`<br> - elu: Returns `tf.keras.layers.ELU`<br> - lrelu: Returns `tf.keras.layers.LeakyReLU`                                                               |
| `padding`     | `str`: One of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. Defaults to "same". |
| `pool_size`   | `int` or tuple of 2 integers: Size of pooling window for the pooling layer. Specifies the height and width of the window. When an int, the height and width are the same. Defaults to (2, 2).                                                   |
| **Returns**   | A `tf.keras.layers` instance encapsulating the block.                                                                                                                                                                                           |

> **Note**: The stride value for the Conv2D layer is always the default value of `1`.

#### <!-- omit in toc --> Example usage

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

### `mesonet.model.fully_connected_layer(ip, units, activation, dropout)`

[[source]](https://github.com/MalayAgarwal-Lee/MesoNet-DeepFakeDetection/blob/a39ffff11bfb2512cb5fca137bd29b9c47d2d54b/mesonet/model.py#L86)

Function to obtain an fully-connected 'block'. A fully-connected block is defined as a set of layers where the first layer is a fully-connected (dense) layer. The entire set of layers is (in this order):

- `tf.keras.layers.Dense` - Fully-connected layer.
- `tf.keras.layers.ReLU`, `tf.keras.layers.ELU` or `tf.keras.layers.LeakyReLU` - Activation layer.
- `tf.keras.layers.Dropout` - Dropout layer.

It feeds the given input to the dense layer and then successively feeds the outputs of one layer to the layer below it, thus encapsulating the entire hidden layer operation applied in the original MesoNet paper and providing a reusable way to replicate the same operation multiple times.

| **Arguments** |                                                                                                                                                                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ip`          | `tf.keras.layers`, Numpy array or list-like: Input for the Dense layer.                                                                                                                                                                |
| `units`       | `int`: Number of neurons in the Dense layer.                                                                                                                                                                                           |
| `activation`  | `str`: Required activation layer. It can be:<br> - relu: Returns `tf.keras.layers.ReLU`<br> - elu: Returns `tf.keras.layers.ELU`<br> - lrelu: Returns `tf.keras.layers.LeakyReLU`<br>The alpha value for the activation is always 0.1. |
| `dropout`     | `float`: Rate of dropout (between 0 and 1) for the Dropout layer.                                                                                                                                                                      |
| **Returns**   | A `tf.keras.layers` instance encapsulating the block.                                                                                                                                                                                  |

#### <!-- omit in toc --> Example usage

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
