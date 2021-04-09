from datetime import datetime
from math import floor, log

from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from .data import get_train_data_generator
from .visualization import plot_loss_curve


def train_model(
    model, train_data_dir, validation_split=None,
    batch_size=32, use_default_augmentation=True,
    augmentations=None, epochs=30, lr=1e-3,
    loss='binary_crossentropy', compile=True,
    lr_decay=True, decay_rate=0.10, decay_limit=1e-6,
    checkpoint=True, stop_early=True, monitor='val_accuracy',
    mode='max', patience=20, tensorboard=True, loss_curve=True
):

    run_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    train_generator, validation_generator = get_train_data_generator(
        train_data_dir=train_data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        use_default_augmentation=use_default_augmentation,
        augmentations=augmentations
    )

    callbacks = []
    if checkpoint:
        filepath = f'run_{run_time}_best_model.hdf5'
        model_checkpoint = ModelCheckpoint(
            filepath, monitor='val_accuracy', verbose=1,
            save_best_only=True
        )
        callbacks.append(model_checkpoint)

    if stop_early:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=patience,
                verbose=1
            )
        )

    if tensorboard:
        log_dir = "logs/fit/" + run_time
        callbacks.append(TensorBoard(
            log_dir, histogram_freq=1, write_images=True))

    if compile:
        if lr_decay:
            num_times = floor(log(decay_limit / lr, decay_rate))
            per_epoch = epochs // num_times
            lr = ExponentialDecay(
                lr,
                decay_steps=(train_generator.samples //
                             batch_size) * per_epoch,
                decay_rate=decay_rate,
                staircase=True,
            )
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = (
        validation_generator.samples // batch_size if validation_generator
        else None
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
