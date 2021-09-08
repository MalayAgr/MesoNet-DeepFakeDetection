---
layout: default
title: Training
nav_order: 3
parent: "mesonet Package"
grand_parent: "Part 1: Model Construction and Training"
---

| **Module** | `train` |
| **Source** | <https://bit.ly/32HZH3R> |
| **Description** | Trains MesoNet variants |
| **Import** | `import mesonet.train` |
| **Depends on** | `mesonet.data`, `mesonet.visualization` |

## Functions

The module has only one function.

### `train_model(model, train_data_dir, validation_split=None, batch_size=32, use_default_augmentation=True, augmentations=None, epochs=30, compile=True, lr=1e-3, loss="binary_crossentropy", lr_decay=True, decay_rate=0.10, decay_limit=1e-6, checkpoint=True, stop_early=True, monitor="val_accuracy", mode="auto", patience=20, tensorboard=True, loss_curve=True)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/e2b2a58f7b4325618f77cc3a24ed67210cc1d62c/mesonet/train.py#L12)

Function to train a model.

This is an all-in-one function which takes a model and a data directory pointing to the training data, and trains the given model on the data after loading it. There are a number of other parameters available to customize the function's behavior.

Most of the times, you'll need to build a model and call this function to train your model, conveniently forgetting about all the other functions available in the package.

| **Arguments**               |                                                                                                                                                                                                                                                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                     | `tf.keras.Model`: Model to be trained.                                                                                                                                                                                                                                                                                        |
| `train_data_dir`            | `str`: Path to the directory containing the training data. See [Note on Directory Structure](./data.md#note-on-directory-structure) for expected directory structure.                                                                                                                                                         |
| `validation_split`          | `float`: Fraction of data to reserve for validation. Should be between 0 and 1. When None or 0.0, there is no reserved data. Defaults to None.                                                                                                                                                                                |
| `batch_size`                | `int`: Size of the batches of the data. Defaults to 32.                                                                                                                                                                                                                                                                       |
| `use_default_augmentations` | `bool`: If True, all augmentations applied in the MesoNet paper are added, in addition to the ones specified in augmentations. See [Note on Augmentations](./data.md#note-on-augmentations). Defaults to True.                                                                                                                |
| `augmentations`             | `dict`: Additional augmentations supported by `ImageDataGenerator`. If an augmentation conflicts with the default augmentations and `use_default_augmentations` is True, the latter takes precedence. Defaults to None.                                                                                                       |
| `epochs`                    | `int`: Number of epochs to train the model. An epoch is an iteration over the entire dataset. Defaults to 30.                                                                                                                                                                                                                 |
| `compile`                   | `bool`: If True, the model is compiled with an optimizer. The optimizer is Adam (with default params). This is useful when the training is stopped and then resumed instead of started for the first time. Set it to False to prevent the optimizer from losing its existing state. Defaults to True.                         |
| `lr`                        | `float`: The (initial) learning rate for the optimizer. Defaults to 1e-3.                                                                                                                                                                                                                                                     |
| `loss`                      | `str`: Loss function for the model. Defaults to 'binary_crossentropy'.                                                                                                                                                                                                                                                        |
| `lr_decay`                  | `bool`: If True, a `ExponentialDecay` schedule is attached to training to gradually decrease the learning rate. Defaults to True.                                                                                                                                                                                             |
| `decay_rate`                | `float`: Rate at which learning rate should decay. Defaults to 0.10.                                                                                                                                                                                                                                                          |
| `decay_limit`               | `float`: Minimum value of the learning rate. It will not decay beyond this point. Defaults to 1e-6. See [Note on Learning Rate Schedule](#note-on-learning-rate-schedule) to learn more about this.                                                                                                                           |
| `checkpoint`                | `bool`: If True, a `ModelCheckpoint` callback is attached to training. The filepath of the saved model is generated using datetime.now(), called as the first line of this function, in the format: %Y/%m/%d-%H-%M-%S. It monitors the validation accuracy and has save_best_only set as True. Defaults to True.              |
| `stop_early`                | `bool`: If True, a `EarlyStopping` callback is attached to training. Defaults to True.                                                                                                                                                                                                                                        |
| `monitor`                   | `str`: The metric to be monitored by the `EarlyStopping` callback. Default to 'val_accuracy'.                                                                                                                                                                                                                                 |
| `mode`                      | `str`: One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity. Defaults to "auto". |
| `patience`                  | `int`: Number of epochs with no improvement after which training will be stopped. Defaults to 20.                                                                                                                                                                                                                             |
| `tensorboard`               | `bool`: If True, a `TensorBoard` callback is attached to training. Defaults to True.                                                                                                                                                                                                                                          |
| `loss_curve`                | `bool`: If True, the training and validation loss are plotted and shown at the end of training. Defaults to True.                                                                                                                                                                                                             |
| **Returns**                 | A `History` instance representing the history of the trained model.                                                                                                                                                                                                                                                           |

#### Example Usage

Here, we use `mesonet.model.build_model()` to obtain the model used in the paper. Assuming that our data is stored in `data/train/`, we train the model on this data, splitting it in a 90-10 ratio for validation. We leave all the other defaults unchanged since they are satisfactory.

```python
from mesonet.model import build_model
from mesonet.train import train_model

model = build_model()
history = train_model(model, 'data/train', validation_split=0.10)
```

It is really that simple to train the architecture in the paper.

## Note on Learning Rate Schedule

Those familiar with `ExponentialDecay` will realize that the above function does not use `decay_steps` as a parameter. Instead, it introduces a new parameter called `decay_limit`, denoting the lowest value the learning rate should decay to. Using this, the `decay_steps` argument for `ExponentialDecay` is calculated as follows:

```python
from math import floor, log

num_times = floor(log(decay_limit / lr, decay_rate))
per_epoch = epochs // num_times
decay_steps = (train_generator.samples // batch_size) * per_epoch
```

Above, `num_times` is the number of times the decay step needs to be applied to reach the limit. For example, if the initial rate is `0.001`, the learning rate is decaying at a rate of `0.10` and the lowest it should go to is `0.000001`, then we need to apply the decay step 3 times. This can be realized as follows:

```python
0.001 * 0.1 = 0.0001
0.0001 * 0.1 = 0.00001
0.00001 * 0.1 = 0.000001
```

`per_epoch` then calculates after how many epochs should a decay step be applied. Continuing with the above example, if we are training for 30 epochs, we need to apply a decay step after every 10 epochs (`30 / 3`) to ensure that the learning rate reaches the limit.

Finally, the `per_epoch` is converted into total number of steps by multiplying it with the number of steps in a single epoch. This becomes the `decay_steps` parameter. Continuing the example, if the dataset has 16,384 images and the default batch size is used, there will be 512 (`16384 / 32`) steps every epoch. Therefore, 10 epochs will have a total of `10 * 512 = 5120` steps. So, a decay step should be applied after every 5120 steps.

The reason behind this is that using a fixed number made the decay either too slow or too fast. This makes it more gradual and was found to yield better models in experiments.

{: .text-center}
[Back to top](#functions)
