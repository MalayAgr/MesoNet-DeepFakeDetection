---
layout: default
title: Data
nav_order: 2
parent: "mesonet Package"
grand_parent: "Part 1: Model Construction and Training"
---

| **Module** | `data` |
| **Source** | <https://bit.ly/32HZH3R> |
| **Description** | Loads data from directories and applies augmentations |
| **Import** | `import mesonet.data` |
| **Depends on** | None |

## <!-- omit in toc --> Jump To

- [Core Functions](#core-functions)
  - {: .fs-3}[`get_train_data_generator(train_data_dir, batch_size, validation_split=None, use_default_augmentation=True, augmentations=None)`](#get_train_data_generatortrain_data_dir-batch_size-validation_splitnone-use_default_augmentationtrue-augmentationsnone)
  - [`get_test_data_generator(test_data_dir, batch_size, shuffle=False)`](#get_test_data_generatortest_data_dir-batch_size-shufflefalse)
- [Helper Functions](#helper-functions)
  - [`_get_datagen(use_default_augmentation=True, **kwargs)`](#_get_datagenuse_default_augmentationtrue-kwargs)
- [Note on Directory Structure](#note-on-directory-structure)
- [Note on Augmentations](#note-on-augmentations)
- [Note on `bloat_data.py`](#note-on-bloat_datapy)

## Core Functions

### `get_train_data_generator(train_data_dir, batch_size, validation_split=None, use_default_augmentation=True, augmentations=None)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/d176ac0173f49bdd178b335d8c8fa62da2b2ad1a/mesonet/data.py#L37)

Function to obtain iterators with data to train a model. The size of the images yielded is determined by `IMG_WIDTH` in `mesonet.utils` and the numbers of channels is always 3 (RGB).

It creates a `ImageDataGenerator` with the the given augmentations and uses the `.flow_from_directory()` method to return a training split and optionally, a validation split.

| **Arguments**              |                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_data_dir`           | `str`: Path to the directory containing training data.                                                                                                                                                                                                                                                                                                                                                                                                    |
| `batch_size`               | `int`: Size of the batches of the data.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `validation_split`         | `float`: Fraction of data to reserve for validation. Should be between 0 and 1. When None or 0.0, there is no reserved data. Defaults to None.                                                                                                                                                                                                                                                                                                            |
| `use_default_augmentation` | `bool`: If True, all augmentations applied in the MesoNet paper are added, in addition to the ones specified in `augmentations`. See [Note on Augmentations](#note-on-augmentations). Defaults to True.                                                                                                                                                                                                                                                   |
| `augmentations`            | `dict`: Additional augmentations supported by `ImageDataGenerator`. If an augmentation conflicts with the default augmentations and `use_default_augmentations` is True, the latter takes precedence. Defaults to None.                                                                                                                                                                                                                                   |
| **Returns**                | A `tuple` (train, val) where:<br>1. If `validation_split` is None or 0.0, train is a `DirectoryIterator` yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape `(batch_size, *target_size, channels)` and y is a numpy array of corresponding labels, and val is None.<br>2. If `validation_split` is between 0 and 1, train is same as above and val is similar to train but yields images in the validation split. |

#### <!-- omit in toc --> Example Usage

Here, we obtain data from a directory called `data/train/`. We obtain a training set and a validation set with with a 80-20 split, in batches of 64 images. In addition to the default augmentations, we also apply `zca_whitening`.

```python
from mesonet.data import get_train_data_generator

train, val = get_train_data_generator('data/train/`, 64, 0.2, augmentations={'zca_whitening': True})
```

### `get_test_data_generator(test_data_dir, batch_size, shuffle=False)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/d176ac0173f49bdd178b335d8c8fa62da2b2ad1a/mesonet/data.py#L104)

Function to obtain an iterator with data to test a model. The size of the images yielded is determined by IMG_WIDTH in utils and the numbers of channels is always 3 (RGB).

This is specifically to obtain data for testing a model. Generally, testing data is not augmented and hence, there is no provision to apply them in the function. Additionally, you can choose to not shuffle the data. This is useful when you want to use the data for generating ROC reports.

| **Arguments**   |                                                                                                                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_data_dir` | `str`: Path to the directory containing test data.                                                                                                                                                      |
| `batch_size`    | `int`: Size of the batches of the data.                                                                                                                                                                 |
| `shuffle`       | `bool`: If True, the shuffle param of `.flow_from_directory()` is set to True so that the order of generating images is random. Set it to False if you wish to obtain a ROC report.                     |
| **Returns**     | A `DirectoryIterator` yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape `(batch_size, *target_size, channels)` and y is a numpy array of corresponding labels. |

#### <!-- omit in toc --> Example Usage

Here, we generate an ROC report on data present in the directory `data/test` on a model saved at `model.hdf5`.

```python
from mesonet.data import get_test_data_generator
from mesonet.model import get_loaded_model, predict, get_classification_report

model = get_loaded_model('model.hdf5')
data = get_test_data_generator('data/test', batch_size=128)

_, preds = predict(model, data)
roc = get_classification_report(data.classes, preds)
```

## Helper Functions

### `_get_datagen(use_default_augmentation=True, **kwargs)`

[[source]](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/d176ac0173f49bdd178b335d8c8fa62da2b2ad1a/mesonet/data.py#L6)

Function to obtain a `ImageDataGenerator` with specified augmentations.

This mainly exists to aide the two functions above. Both of them use this to obtain a `ImageDataGenerator` instance with the provided augmentations, where applicable. See [Note on Augmentations](#note-on-augmentations) for information on the augmentations.

You'll probably not need to use this.

| **Arguments**              |                                                                                                                                                                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_default_augmentation` | `bool`: If True, all augmentations applied in the MesoNet paper are added, in addition to the ones specified in `augmentations`. See [Note on Augmentations](#note-on-augmentations). Defaults to True.                 |
| `**kwargs`                 | `dict`: Additional augmentations supported by `ImageDataGenerator`. If an augmentation conflicts with the default augmentations and `use_default_augmentations` is True, the latter takes precedence. Defaults to None. |
| **Returns**                | An `ImageDataGenerator` object with specified augmentations.                                                                                                                                                            |

## Note on Directory Structure

Both the core functions expect a string parameter denoting the path to the directory where data is stored. The expected directory structure for both of them is as follows:

```bash
└── data/
    ├── real/
    │   ├── img1.png
    │   └── img2.png
    └── forged/
        ├── img1.png
        └── img2.png
```

## Note on Augmentations

By default, `get_train_data_generator()` uses the augmentations applied in the original paper. These augmentations are passed to `ImageDataGenerator`. They are as follows:

- `rotation_range`: `15`
- `zoom_range`: `0.2`
- `brightness_range`: `(0.8, 1.2)`
- `channel_shift_range`: `30`
- `horizontal_flip`: `True`

It is possible to override these values or add more augmentations of your own. Also, note that a rescaling by 255 is always applied to the images (`rescale=1.0 / 255`).

## Note on `bloat_data.py`

In my experiments, I found that the [dataset](https://my.pcloud.com/publink/show?code=XZLGvd7ZI9LjgIy7iOLzXBG5RNJzGFQzhTRy) used has too many images in the validation split. Almost 37% of the total data has been reserved for testing. You can take a look at the distribution [here](https://github.com/MalayAgr/MesoNet-DeepFakeDetection#23-the-data).

> **Note**: In today's jargon, what the paper refers to as the validation split is called the test split.

In case you come to the same conclusions, the project comes with a [script](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/blob/main/bloat_data.py) which takes all the images across both the splits and creates a new dataset, where only a small portion (by default, 10%) of the data is reserved for testing.

The original dataset has the following structure:

```bash
└── data/
    ├── train/
    │   ├── real/
    │   │   ├── img1.png
    │   │   └── img2.png
    │   └── df/
    │       ├── img1.png
    │       └── img2.png
    └── validation/
        ├── real/
        │   ├── img1.png
        │   └── img2.png
        └── df/
            ├── img1.png
            └── img2.png
```

Given this structure, using the script is simple. You'll notice that it defines 5 global variables. You can change the values of these variables to configure the script.

- `DATA_DIR` - This should be a `str` containing the path to the directory where the original dataset is stored. Note that it should be the top-level directory. This means that, in the above structure, this variable should be `data/`.
- `TARGET_DIR` - This should be a `str` containing the path to the directory where you want the script to create your new dataset. Don't worry about the directory not existing because the script handles that for you.
- `REAL_DIR` - This is the name of the sub-directory in your original dataset where images belonging to the real class are stored. For example, if your real images are in `data/train/real`, then this should be `real`. Note that the name of the directory should be the same in both the train and validation split.
- `FAKE_DIR` - This is the same as `REAL_DIR` except it refers to the deepfake class.
- `PROP` - This is the proportion of the data that should be in test split of the new dataset.

Once you've set these variables, you can run the script as:

```bash
$ python bloat_data.py
Creating directories at new_data/....
Found 19456 images in deepfake_database/....
Copying 17511 files to new_data/train/....
Copying 1945 files to /new_data/test/....
The training set will have 10376 real images and 7135 fake images....
The test set will have 1132 real images and 813 fake images...
Try again? [Yy/Nn]
```

The script will ask you whether you want to repeat the process. If yes, it will undo it's work and again randomly split the dataset. This allows you to repeat the process until you achieve the desired split.

{: .text-center}
[Back to top](#-jump-to)
