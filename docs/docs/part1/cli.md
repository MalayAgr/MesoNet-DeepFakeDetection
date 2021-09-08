---
layout: default
title: "Command Line Interface (CLI)"
nav_order: 5
parent: "Part 1: Model Construction and Training"
has_toc: false
---

| **Package** | `cli` |
| **Source** | <https://bit.ly/3nydEe5> |
| **Description** | Command-line interface for MesoNet variants. |
| **Depends on** | `mesonet.train`, `mesonet.model` |

## <!-- omit in toc --> Table of Contents

- [Introduction](#introduction)
- [Training Sub-Command](#training-sub-command)
  - [Usage](#usage)
  - [Arguments and Options](#arguments-and-options)
    - [Data Directory, Augmentations and Validation Split](#data-directory-augmentations-and-validation-split)
    - [Epochs and Batch Size](#epochs-and-batch-size)
    - [Compilation, Learning Rate and Learning Rate Decay](#compilation-learning-rate-and-learning-rate-decay)
    - [Early Stopping](#early-stopping)
    - [Model Checkpoint, TensorBoard and Loss Curve](#model-checkpoint-tensorboard-and-loss-curve)
  - [Examples](#examples)
- [Prediction Sub-Command](#prediction-sub-command)
  - [Usage](#usage-1)
  - [Arguments and Options](#arguments-and-options-1)
    - [Positional Arguments](#positional-arguments)
    - [Optional Arguments](#optional-arguments)
  - [Example](#example)

## Introduction

The project comes equipped with a command line interface (CLI) to train and obtain predictions from MesoNet variants, built using `argparse`.

The entrypoint for the CLI is the [`mesonet.py`](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/mesonet.py) file. You can use the CLI with the following command:

```bash
python mesonet.py [sub-command]
```

There are two sub-commands available:

- `train` - Sub-command responsible for training a MesoNet variant. Currently, it only supports training the architecture detailed in the original paper.
- `predict` - Sub-command responsible for obtaining predictions on data from a saved MesoNet variant. You can use both an HDF5 file and a directory containing a `.pb` file.

## Training Sub-Command

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/cli/training_cli.py)

The `train` sub-command is responsible for training a MesoNet variant. Currently, it only supports training the architecture detailed in the original paper.

### Usage

```bash
python mesonet.py train
    [-h] [-nda] [-s SPLIT] [-e EPOCHS]
    [-bs BS] [-nc] [-lo LOSS] [-lr LR]
    [-nld] [-dr RATE] [-dl LIMIT] [-nse]
    [-m MONITOR] [--mode {max,min,auto}] [-pa PATIENCE] [-ncp]
    [-ntb] [-np] path
```

### Arguments and Options

Run the following command to get the help text:

```bash
python mesonet.py train --help
```

#### Data Directory, Augmentations and Validation Split

- `path`: Path to the directory containing the training data. See [Note on Directory Structure](../part1/mesonet/data.md#note-on-directory-structure) for the expected directory structure.
- `-nda`, `--no-def-aug`: Disable the default augmentations used in the paper. See [Note on Augmentations](../part1/mesonet/data.md#note-on-augmentations) for a list of the augmentations applied.
- `-s SPLIT`, `--validation-split SPLIT`: Size of the validation split as a proportion (between 0 and 1). Defaults to `None`.

#### Epochs and Batch Size

- `-e EPOCHS`, `--epochs EPOCHS`: Number of epochs the model should be trained for. Defaults to 30.
- `-bs BS`, `--batch-size BS`: Batch size for the dataset. Defaults to 32.

#### Compilation, Learning Rate and Learning Rate Decay

- `-nc`, `--no-compile`: Disable model compilation with an optimizer (useful for resuming training). For details on which optimizer is used, see the `compile` argument in [Training](../part1/mesonet/train.md).
- `-lo LOSS`, `--loss LOSS`: Loss function. Defaults to `binary_crossentropy`.
- `-lr LR`, `--learning-rate LR`: Learning rate (initial) of the model. Defaults to `0.001`.
- `-nld`, `--no-decay`: Disable learning rate decay.
- `-dr RATE`, `--decay-rate RATE`: Rate at which the learning rate should decay. Defaults to `0.1`.
- `-dl LIMIT`, `--decay-limit LIMIT`: Learning rate value after which decay should stop. Defaults to `1e-06`. See [Note on Learning Rate Schedule](../part1/mesonet/train.md#note-on-learning-rate-schedule) to learn more about how this is used.

#### Early Stopping

- `-nse`, `--no-stop-early`: Disable EarlyStopping callback.
- `-m MONITOR`, `--monitor MONITOR`: Metric to monitor for the EarlyStopping callback. Defaults to `val_accuracy`.
- `--mode {max,min,auto}`: See <https://keras.io/api/callbacks/early_stopping/>. Defaults to `max`.
- `-pa PATIENCE`, `--patience PATIENCE`: Number of epochs with no improvement after which training will be stopped. Defaults to `20`.

#### Model Checkpoint, TensorBoard and Loss Curve

- `-ncp`, `--no-checkpoint`: Disable ModelCheckpoint callback.
- `-ntb`, `--no-tensorboard`: Disable TensorBoard callback.
- `-np`, `--no-plot`: Disable plotting of loss curve.

### Examples

In all the examples below, data is assumed to be at `data/train/`.

- Training with all default options:

  ```bash
  python mesonet.py train data/train/
  ```

- Training for 50 epochs with a batch size of 64:

  ```bash
  python mesonet.py train -e 50 -bs 64 data/train/
  ```

- Turning off TensorBoard:

  ```bash
  python mesonet.py train -ntb data/train/
  ```

- Training for 50 epochs with a batch size of 64, initial learning rate of 0.0001 and decay of 0.25:

  ```bash
  python mesonet.py train -e 50 -bs 64 -lr 1e-4 -dr 0.25 data/train/
  ```

## Prediction Sub-Command

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/cli/prediction_cli.py)

The `predict` sub-command is responsible for obtaining predictions on data from a saved MesoNet variant. You can use both an HDF5 file and a directory containing a `.pb` file. Note that this only works when the output layer has a sigmoid activation.

### Usage

```bash
python mesonet.py predict [-h] [-t THRESHOLD] [-p] [-cl] model data_dir
```

### Arguments and Options

Run the following command to get the help text:

```bash
python mesonet.py predict --help
```

#### Positional Arguments

- `model`: Path to the file/directory containing the trained MesoNet variant.
- `data_dir`: Path to the directory containing the dataset. Contrary to `train` sub-command, it isn't necessary for your directory to have any specific structure.

#### Optional Arguments

- `-t THRESHOLD`, `--threshold THRESHOLD`: Probability threshold for classification. Images having sigmoid probabilities equal to or above this value will be classified into the positive class. Defaults to 0.5.
- `-p, --include-probabilites`: Show predicted probabilities along with labels. Note that the probability is always for the image belonging to the predicted class and not for the image belonging to the positive class.
- `-cl`, `--classification-report`: Generate the classification report for the dataset. Note that to be able to use this option, the data directory does need to have the same structure as that mentioned in [Note on Directory Structure](../part1/mesonet/data.md#note-on-directory-structure).

### Example

Assuming data is in `data/predict/` and using one of the trained models provided in `trained_models`, below we obtain predictions, enabling the probabilities and classification report options:

```bash
$ python mesonet.py predict -p -cl trained_models/model1_18epochs_valacc0.9252.hdf5 data/predict
Found 55 images belonging to 2 classes.
1/1 [==============================] - 2s 2s/step
Classification report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        43
           1       1.00      1.00      1.00        12

    accuracy                           1.00        55
   macro avg       1.00      1.00      1.00        55
weighted avg       1.00      1.00      1.00        55

###############################################
| Filenames            | Predictions   |   Probability |
|----------------------|---------------|---------------|
| deepfake/114_132.jpg | Deepfake      |       99.5502 |
| deepfake/114_144.jpg | Deepfake      |       99.9737 |
| deepfake/114_150.jpg | Deepfake      |       99.6843 |
| deepfake/114_186.jpg | Deepfake      |       99.9748 |
| deepfake/114_204.jpg | Deepfake      |       99.9995 |
| deepfake/114_228.jpg | Deepfake      |       99.9984 |
| deepfake/114_240.jpg | Deepfake      |       99.9991 |
| deepfake/114_246.jpg | Deepfake      |       99.9001 |
| deepfake/114_264.jpg | Deepfake      |       99.9998 |
| deepfake/114_276.jpg | Deepfake      |       99.9985 |
| deepfake/114_288.jpg | Deepfake      |       77.8174 |
| deepfake/114_300.jpg | Deepfake      |       99.9996 |
| deepfake/114_306.jpg | Deepfake      |       99.9999 |
| deepfake/114_324.jpg | Deepfake      |       99.9998 |
| deepfake/114_348.jpg | Deepfake      |      100.0000 |
| deepfake/114_354.jpg | Deepfake      |      100.0000 |
| deepfake/114_390.jpg | Deepfake      |       99.9987 |
| deepfake/114_396.jpg | Deepfake      |      100.0000 |
| deepfake/114_402.jpg | Deepfake      |      100.0000 |
| deepfake/114_408.jpg | Deepfake      |      100.0000 |
| deepfake/114_414.jpg | Deepfake      |       99.9996 |
| deepfake/114_426.jpg | Deepfake      |       99.9993 |
| deepfake/114_438.jpg | Deepfake      |       99.9993 |
| deepfake/114_450.jpg | Deepfake      |       99.9990 |
| deepfake/114_510.jpg | Deepfake      |       99.9999 |
| deepfake/114_516.jpg | Deepfake      |       99.9999 |
| deepfake/114_534.jpg | Deepfake      |       99.9999 |
| deepfake/114_540.jpg | Deepfake      |       99.9997 |
| deepfake/114_564.jpg | Deepfake      |       99.9965 |
| deepfake/114_570.jpg | Deepfake      |       99.9991 |
| deepfake/114_576.jpg | Deepfake      |       99.9998 |
| deepfake/114_588.jpg | Deepfake      |       99.9982 |
| deepfake/114_618.jpg | Deepfake      |       99.9999 |
| deepfake/114_642.jpg | Deepfake      |       99.9999 |
| deepfake/114_648.jpg | Deepfake      |      100.0000 |
| deepfake/114_666.jpg | Deepfake      |       99.9807 |
| deepfake/114_672.jpg | Deepfake      |       99.9989 |
| deepfake/114_702.jpg | Deepfake      |       99.9999 |
| deepfake/114_708.jpg | Deepfake      |      100.0000 |
| deepfake/114_714.jpg | Deepfake      |       99.9998 |
| deepfake/114_720.jpg | Deepfake      |       99.9995 |
| deepfake/114_726.jpg | Deepfake      |       99.9999 |
| deepfake/114_738.jpg | Deepfake      |      100.0000 |
| real/34_102.jpg      | Real          |       75.8813 |
| real/34_108.jpg      | Real          |       85.5676 |
| real/34_114.jpg      | Real          |       95.6834 |
| real/34_120.jpg      | Real          |       71.2622 |
| real/34_54.jpg       | Real          |       97.7189 |
| real/34_60.jpg       | Real          |       98.9342 |
| real/34_66.jpg       | Real          |       99.0665 |
| real/34_72.jpg       | Real          |       99.5065 |
| real/34_78.jpg       | Real          |       99.8474 |
| real/34_84.jpg       | Real          |       95.4600 |
| real/34_90.jpg       | Real          |       92.4506 |
| real/34_96.jpg       | Real          |       97.2753 |
```
