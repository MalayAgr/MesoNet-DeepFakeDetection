import argparse

from model import build_model
from train import train_model


def add_train_subparser(subparsers):
    train_parser = subparsers.add_parser(
        'train',
        description='Train the MesoNet architecture as detailed in the paper.',
    )

    data_group = train_parser.add_argument_group(
        'data directory, augmentations and validation split',
    )

    data_group.add_argument(
        'path',
        help='Path to the directory containing the training data',
    )

    data_group.add_argument(
        '-nda', '--no-def-aug',
        action='store_false',
        help='Disable the default augmentations used in the paper',
        dest='use_default_augmentation',
    )

    data_group.add_argument(
        '-s', '--validation-split',
        type=float,
        default=None,
        metavar='SPLIT',
        help='The size of the validation split as a proportion (default: %(default)s)'
    )

    epoch_group = train_parser.add_argument_group(
        'epochs, batch size'
    )

    epoch_group.add_argument(
        '-e', '--epochs',
        type=int,
        default=30,
        help='Number of epochs the model should be trained for (default: %(default)s)'
    )

    epoch_group.add_argument(
        '-bs', '--batch-size',
        type=int,
        default=32,
        metavar='BS',
        help='The batch size for the dataset (default: %(default)s)'
    )

    optimizer_group = train_parser.add_argument_group(
        'compilation, learning rate and learning rate decay',
    )

    optimizer_group.add_argument(
        '-nc', '--no-compile',
        action='store_false',
        help='Disable model compilation (useful for resuming training)',
        dest='compile'
    )

    optimizer_group.add_argument(
        '-lo', '--loss',
        default='binary_crossentropy',
        help="The loss function (default: %(default)s)"
    )

    optimizer_group.add_argument(
        '-lr', '--learning-rate',
        type=float,
        default=1e-3,
        metavar='LR',
        help="The (initial) learning rate of the model (default: %(default)s)",
        dest='lr'
    )

    optimizer_group.add_argument(
        '-nld', '--no-decay',
        action='store_false',
        help='Disable learning rate decay',
        dest='lr_decay',
    )

    optimizer_group.add_argument(
        '-dr', '--decay-rate',
        type=float,
        default=1e-1,
        metavar='RATE',
        help="Rate at which the learning rate should decay (default: %(default)s)"
    )

    optimizer_group.add_argument(
        '-dl', '--decay-limit',
        type=float,
        default=1e-6,
        metavar='LIMIT',
        help="The learning rate value after which decay should stop (default: %(default)s)"
    )

    stop_early_group = train_parser.add_argument_group(
        'stop early, monitor, mode, patience'
    )

    stop_early_group.add_argument(
        '-nse', '--no-stop-early',
        action='store_false',
        help='Disable EarlyStopping callback',
        dest='stop_early'
    )

    stop_early_group.add_argument(
        '-m', '--monitor',
        default='val_accuracy',
        help="The metric to monitor for the EarlyStopping callback (default: %(default)s)"
    )

    stop_early_group.add_argument(
        '--mode',
        default='max',
        choices=['max', 'min', 'auto'],
        help=(
            "See https://keras.io/api/callbacks/early_stopping/"
            "(default: %(default)s)"
        )
    )

    stop_early_group.add_argument(
        '-pa', '--patience',
        type=int,
        default=20,
        help=(
            'Number of epochs with no improvement after which '
            'training will be stopped (default: %(default)s)'
        )
    )

    misc_group = train_parser.add_argument_group(
        'checkpoint, tensorboard, plot'
    )

    misc_group.add_argument(
        '-ncp', '--no-checkpoint',
        action='store_false',
        help='Disabe ModelCheckpoint callback',
        dest='checkpoint',
    )

    misc_group.add_argument(
        '-ntb', '--no-tensorboard',
        action='store_false',
        help='Disable TensorBoard callback',
        dest='tensorboard',
    )

    misc_group.add_argument(
        '-np', '--no-plot',
        action='store_false',
        help='Disable plotting of loss curve',
        dest='loss_curve'
    )

    return train_parser


def execute_train(args):
    args = vars(args)
    args['train_data_dir'] = args['path']
    args.pop('func', None)
    args.pop('path', None)

    model = build_model()

    return train_model(model=model, **args)


def arg_parser():
    parser = argparse.ArgumentParser(
        prog='mesonet',
        description=(
            'A command-line interface for MesoNet, a Deepfake Detector.\n'
            'Original paper: https://arxiv.org/abs/1809.00888.\n'
            'Train, make predictions and visualize MesoNet.\n'
        )
    )

    subparsers = parser.add_subparsers(help='Sub-commands')
    train_parser = add_train_subparser(subparsers)
    train_parser.set_defaults(func=execute_train)

    args = parser.parse_args()
    return args.func(args)
