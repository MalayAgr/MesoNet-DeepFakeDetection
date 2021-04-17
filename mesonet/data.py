from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .utils import IMG_WIDTH


def _get_datagen(use_default_augmentation=True, **kwargs):
    """
    Function to obtain a ImageDataGenerator with specified augmentations.

    Args:
        use_default_augmentation (bool): If True, all augmentations applied in the
            MesoNet paper are added, in addition to the ones specified in **kwargs.
            See https://github.com/DariusAf/MesoNet/issues/4#issuecomment-448694527.
            Note that rescaling by 255 is always added.
            Defaults to True.
        **kwargs (dict): Additional augmentations supported by ImageDataGenerator.
            If an augmentation conflicts with the default augmentations and
            use_default_augmentations is True, the latter takes precedence.

    Returns:
        An `ImageDataGenerator` object with specified augmentations
    """
    kwargs.update({"rescale": 1.0 / 255})
    if use_default_augmentation:
        kwargs.update(
            {
                "rotation_range": 15,
                "zoom_range": 0.2,
                "brightness_range": (0.8, 1.2),
                "channel_shift_range": 30,
                "horizontal_flip": True,
            }
        )
    return ImageDataGenerator(**kwargs)


def get_train_data_generator(
    train_data_dir,
    batch_size,
    validation_split=None,
    use_default_augmentation=True,
    augmentations=None,
):
    """
    Function to obtain iterators with data to train a model.
    The size of the images yielded is determined by IMG_WIDTH in utils
    And the numbers of channels is always 3 (RGB).

    Args:
        train_data_dir (str): Path to the directory containing training data.
        batch_size (int): Size of the batches of the data.
        validation_split (float): Fraction of data to reserve for validation.
            Should be between 0 and 1. When None or 0.0, there is no reserved data.
            Defaults to None.
        use_default_augmentation (bool): If True, all augmentations applied in the
            MesoNet paper are added, in addition to the ones specified in augmentations.
            See https://github.com/DariusAf/MesoNet/issues/4#issuecomment-448694527.
            Defaults to True.
        augmentations (dict): Additional augmentations supported by ImageDataGenerator.
            If an augmentation conflicts with the default augmentations and
            use_default_augmentations is True, the latter takes precedence.
            Defaults to None.

    Returns:
        A tuple in the format (train, val) where:
            1. If validation_split is None or 0.0, train is a DirectoryIterator yielding
                tuples of (x, y) where x is a numpy array containing a batch of
                images with shape (batch_size, *target_size, channels) and y is a
                numpy array of corresponding labels, and val is None.
            2. Otherwise, train is same as above and val is similar to train but
                yields images in the validation split.
    """
    if not augmentations:
        augmentations = {}

    train_datagen = _get_datagen(
        use_default_augmentation=use_default_augmentation,
        validation_split=validation_split if validation_split else 0.0,
        **augmentations
    )

    train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
    )

    validation_generator = None

    if validation_split is None or int(validation_split) == 0:
        validation_generator = train_datagen.flow_from_directory(
            directory=train_data_dir,
            target_size=(IMG_WIDTH, IMG_WIDTH),
            batch_size=batch_size,
            class_mode="binary",
            subset="validation",
        )

    return train_generator, validation_generator


def get_test_data_generator(test_data_dir, batch_size, shuffle=False):
    """
    Function to obtain an iterator with data to test a model.
    The size of the images yielded is determined by IMG_WIDTH in utils
    And the numbers of channels is always 3 (RGB).

    Args:
        test_data_dir (str): Path to the directory containing test data.
        batch_size (int): Size of batches of the data.
        shuffle (bool, optional): If True, the shuffle param of .flow_from_directory
            is set to True so that the order of generating images is random.
            Set it to False if you wish to obtain a ROC report.

    Returns:
        A DirectoryIterator yielding tuples of (x, y) where x is a numpy
        array containing a batch of images with
        shape (batch_size, *target_size, channels) and y is a numpy array
        of corresponding labels.
    """
    test_datagen = _get_datagen(use_default_augmentation=False)
    return test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=(IMG_WIDTH, IMG_WIDTH),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=shuffle,
    )
