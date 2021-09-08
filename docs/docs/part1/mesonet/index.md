---
layout: default
title: "mesonet Package"
nav_order: 1
parent: "Part 1: Model Construction and Training"
has_children: true
has_toc: false
---

The `mesonet` package provides functions to build, train, obtain predictions from and visualize MesoNet variants.

It has been divided into four main modules:

- [`model`](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/mesonet/model.py) - This module provides functions to build and obtain predictions from MesoNet variants. It also consists of helper functions such as obtaining a classification report, saving the model to a file/directory, saving model history, etc.
- [`data`](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/mesonet/data.py) - This module provides functions to apply augmentations and load images from the dataset for training and testing.
- [`train`](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/mesonet/train.py) - This module provides a single function to train a MesoNet variant.
- [`visualization`](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/mesonet/visualization.py) - This module provides functions to visualize loss curves and the intermediate convolutional layers.

There is also a [`utils`](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/mesonet/utils.py) module for the purpose of storing any miscellaneous utilities required in the package. Currently, it only stores the input size of images as the constant `IMG_WIDTH`. The default value is `256`. Therefore, by default, any model you train will use `256 x 256` images as its input. You can easily change this by modifying the constant.
