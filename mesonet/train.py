from datetime import datetime
from math import floor, log

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from .data import get_train_data_generator
from .visualization import plot_loss_curve


def train_model(
    model,
    train_data_dir,
    validation_split=None,
    batch_size=32,
    use_default_augmentation=True,
    augmentations=None,
    epochs=30,
    compile=True,
    lr=1e-3,
    loss="binary_crossentropy",
    lr_decay=True,
    decay_rate=0.10,
    decay_limit=1e-6,
    checkpoint=True,
    stop_early=True,
    monitor="val_accuracy",
    mode="auto",
    patience=20,
    tensorboard=True,
    loss_curve=True,
):
    """
    Function to train a model.

    Args:
        model (tf.keras.Model): model to be trained.

        train_data_dir (str): Path to the directory containing the training data.

        validation_split (float): Fraction of data to reserve for validation.
            Should be between 0 and 1. When None or 0.0, there is no reserved data.
            Defaults to None.

        batch_size (int): Size of the batches of the data. Defaults to 32.

        use_default_augmentation (bool): If True, all augmentations applied in the
            MesoNet paper are added, in addition to the ones specified in augmentations.
            See https://github.com/DariusAf/MesoNet/issues/4#issuecomment-448694527.
            Defaults to True.

        augmentations (dict): Additional augmentations supported by ImageDataGenerator.
            If an augmentation conflicts with the default augmentations and
            use_default_augmentations is True, the latter takes precedence.
            Defaults to None.

        epochs (int): Number of epochs to train the model. An epoch is
            an iteration over the entire dataset. Defaults to 30.

        compile (bool): If True, the model is compiled with an optimizer.
            The optimizer is Adam (with default params). This is useful when
            the training is stopped and then resumed instead of started for the
            first time. Set it to False to prevent the optimizer from losing its
            existing state. Defaults to True.

        lr (float): The (initial) learning rate for the optimizer. Defaults to 1e-3.

        loss (str, optional): The loss function for the model.
            Defaults to 'binary_crossentropy'.

        lr_decay (bool): If True, a ExponentialDecay schedule is attached to training
            to gradually decrease the learning rate. Defaults to True.

        decay_rate (float): Rate at which learning rate should decay.
            Defaults to 0.10.

        decay_limit (float): Minimum value of the learning rate. It will not decay
            beyond this point. Defaults to 1e-6. Using this, the decay_steps
            argument of ExponentialDecay is calculated as:
                num_times = floor(log(decay_limit / lr, decay_rate))
                per_epoch = epochs // num_times
                decay_steps = (train_generator.samples // batch_size) * per_epoch,
            where:
                num_times = Number of times decay needs to be applied during the course
                    of training.
                per_epoch = Number of epochs after which one step of decay should be applied.
                decay_steps = per_epoch converted into number of steps.
            In experiments, it was found that this approach yields a generally better model
            than manually setting the decay_steps.

        checkpoint (bool): If True, a ModelCheckpoint callback is attached to training.
            The filepath of the saved model is generated using datetime.now(), called as the
            first line of this function, in the format: %Y/%m/%d-%H-%M-%S. It monitors the
            validation accuracy and has save_best_only set as True. Defaults to True.

        stop_early (bool): If True, a EarlyStopping callback is attached to training.
            Defaults to True.

        monitor (str): The metric to be monitored by the EarlyStopping callback.

        mode (str): One of {"auto", "min", "max"}. In min mode, training will stop when
            the quantity monitored has stopped decreasing; in "max" mode it will stop when
            the quantity monitored has stopped increasing; in "auto" mode, the direction
            is automatically inferred from the name of the monitored quantity. Defaults to "auto".

        patience (int): Number of epochs with no improvement after which training will
            be stopped. Defaults to 20.

        tensorboard (bool, optional): If True, a TensorBoard callback is attached to training.
            Defaults to True.

        loss_curve (bool): If True, the training and validation loss are plotted and shown
            at the end of training. Defaults to True.

    Returns:
        A History instance representing the history of the model.
    """
    run_time = datetime.now().strftime("%Y/%m/%d-%H-%M-%S")

    train_generator, validation_generator = get_train_data_generator(
        train_data_dir=train_data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        use_default_augmentation=use_default_augmentation,
        augmentations=augmentations,
    )

    callbacks = []
    if checkpoint:
        filepath = f"run_{run_time}_best_model.hdf5"
        model_checkpoint = ModelCheckpoint(
            filepath, monitor="val_accuracy", verbose=1, save_best_only=True
        )
        callbacks.append(model_checkpoint)

    if stop_early:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=1)
        )

    if tensorboard:
        log_dir = "logs/fit/" + run_time
        callbacks.append(TensorBoard(log_dir, histogram_freq=1, write_images=True))

    if compile:
        if lr_decay:
            num_times = floor(log(decay_limit / lr, decay_rate))
            per_epoch = epochs // num_times
            lr = ExponentialDecay(
                lr,
                decay_steps=(train_generator.samples // batch_size) * per_epoch,
                decay_rate=decay_rate,
                staircase=True,
            )
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = (
        validation_generator.samples // batch_size if validation_generator else None
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    if loss_curve:
        plot_loss_curve(history)

    return history
