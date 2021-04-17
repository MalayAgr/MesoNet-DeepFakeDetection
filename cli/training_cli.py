from mesonet.model import build_model
from mesonet.train import train_model


def execute_train(args):
    args = vars(args)
    args["train_data_dir"] = args["path"]
    args.pop("func", None)
    args.pop("path", None)

    model = build_model()

    return train_model(model=model, **args)


def train_parser(subparser):
    parser = subparser.add_parser(
        "train",
        description="Train the MesoNet architecture as detailed in the paper.",
    )

    data_group = parser.add_argument_group(
        "data directory, augmentations and validation split",
    )

    data_group.add_argument(
        "path",
        help="path to the directory containing the training data",
    )

    data_group.add_argument(
        "-nda",
        "--no-def-aug",
        action="store_false",
        help="disable the default augmentations used in the paper",
        dest="use_default_augmentation",
    )

    data_group.add_argument(
        "-s",
        "--validation-split",
        type=float,
        default=None,
        metavar="SPLIT",
        help="size of the validation split as a proportion (default: %(default)s)",
    )

    epoch_group = parser.add_argument_group("epochs, batch size")

    epoch_group.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=30,
        help="#epochs the model should be trained for (default: %(default)s)",
    )

    epoch_group.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=32,
        metavar="BS",
        help="batch size for the dataset (default: %(default)s)",
    )

    optimizer_group = parser.add_argument_group(
        "compilation, learning rate and learning rate decay",
    )

    optimizer_group.add_argument(
        "-nc",
        "--no-compile",
        action="store_false",
        help="disable model compilation (useful for resuming training)",
        dest="compile",
    )

    optimizer_group.add_argument(
        "-lo",
        "--loss",
        default="binary_crossentropy",
        help="loss function (default: %(default)s)",
    )

    optimizer_group.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        metavar="LR",
        help="(initial) learning rate of the model (default: %(default)s)",
        dest="lr",
    )

    optimizer_group.add_argument(
        "-nld",
        "--no-decay",
        action="store_false",
        help="disable learning rate decay",
        dest="lr_decay",
    )

    optimizer_group.add_argument(
        "-dr",
        "--decay-rate",
        type=float,
        default=1e-1,
        metavar="RATE",
        help="rate at which the learning rate should decay (default: %(default)s)",
    )

    optimizer_group.add_argument(
        "-dl",
        "--decay-limit",
        type=float,
        default=1e-6,
        metavar="LIMIT",
        help="learning rate value after which decay should stop (default: %(default)s)",
    )

    stop_early_group = parser.add_argument_group("stop early, monitor, mode, patience")

    stop_early_group.add_argument(
        "-nse",
        "--no-stop-early",
        action="store_false",
        help="disable EarlyStopping callback",
        dest="stop_early",
    )

    stop_early_group.add_argument(
        "-m",
        "--monitor",
        default="val_accuracy",
        help="metric to monitor for the EarlyStopping callback (default: %(default)s)",
    )

    stop_early_group.add_argument(
        "--mode",
        default="max",
        choices=["max", "min", "auto"],
        help=(
            "see https://keras.io/api/callbacks/early_stopping/"
            "(default: %(default)s)"
        ),
    )

    stop_early_group.add_argument(
        "-pa",
        "--patience",
        type=int,
        default=20,
        help=(
            "number of epochs with no improvement after which "
            "training will be stopped (default: %(default)s)"
        ),
    )

    misc_group = parser.add_argument_group("checkpoint, tensorboard, plot")

    misc_group.add_argument(
        "-ncp",
        "--no-checkpoint",
        action="store_false",
        help="disabe ModelCheckpoint callback",
        dest="checkpoint",
    )

    misc_group.add_argument(
        "-ntb",
        "--no-tensorboard",
        action="store_false",
        help="disable TensorBoard callback",
        dest="tensorboard",
    )

    misc_group.add_argument(
        "-np",
        "--no-plot",
        action="store_false",
        help="disable plotting of loss curve",
        dest="loss_curve",
    )

    parser.set_defaults(func=execute_train)

    return parser
