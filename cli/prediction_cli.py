from mesonet.model import make_prediction
from tabulate import tabulate


def execute_predict(args):
    args = vars(args)

    args["model_path"] = args["model"]

    args.pop("model", None)
    args.pop("func", None)

    preds, report = make_prediction(**args)

    if report:
        print("Classification report:")
        print(report)
        print("###############################################")

    headers = ["Filenames", "Predictions"]
    if preds.shape[-1] == 3:
        headers += ["Probability"]

    print(tabulate(preds, headers=headers, tablefmt="github", floatfmt=".4f"))


def predict_parser(subparser):
    parser = subparser.add_parser(
        "predict",
        description="Make predictions using MesoNet and visualize results.",
    )

    parser.add_argument(
        "model", help="path to the file/directory containing the trained model"
    )

    parser.add_argument("data_dir", help="path to the directory containing the dataset")

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="probability threshold for classification (default: %(default)s)",
    )

    # TODO: Add visualization
    # parser.add_argument(
    #     '-vl', '--visualize-layers',
    #     default=None,
    #     metavar='idx',
    #     help='0-indexed indices of conv layers to be visualized (eg: 03)',
    #     dest='conv_idx'
    # )

    parser.add_argument(
        "-p",
        "--include-probabilites",
        action="store_true",
        help="show predicted probabilities along with labels",
        dest="return_probs",
    )

    parser.add_argument(
        "-cl",
        "--classification-report",
        action="store_true",
        help="generate the classification report for the dataset",
        dest="return_report",
    )

    parser.set_defaults(func=execute_predict)

    return parser
