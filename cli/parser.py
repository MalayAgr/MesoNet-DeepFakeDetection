import argparse

from .training_cli import train_parser
from .prediction_cli import predict_parser


def arg_parser():
    parser = argparse.ArgumentParser(
        prog="mesonet",
        description=(
            "A command-line interface for MesoNet, a Deepfake Detector.\n"
            "Original paper: https://arxiv.org/abs/1809.00888.\n"
            "Train, make predictions and visualize MesoNet.\n"
        ),
    )

    subparser = parser.add_subparsers(help="Sub-commands")
    _ = train_parser(subparser)
    _ = predict_parser(subparser)

    args = parser.parse_args()
    args.func(args)
