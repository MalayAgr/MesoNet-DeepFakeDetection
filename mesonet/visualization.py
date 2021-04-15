import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from .model import get_activation_model


def plot_loss_curve(history):
    """
    Function to plot training and validation loss of a trained model.

    Args:
        history (`History` object): The history of the model.
    """
    plt.plot(history.history["loss"], "r", label="train")
    plt.plot(history.history["val_loss"], "g", label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def _visualize_conv_layers_single_img(
    activations,
    conv_idx,
):
    """
    Function to visualize output of multiple conv layers for a single image.

    Args:
        activations (Numpy array): The computed outputs of conv layers for a image
        conv_idx (List-like): The indices of the conv layers.
    """
    images_per_row = 4

    for activation, idx in zip(activations, conv_idx):
        num_filters = activation.shape[-1]

        imgs = [activation[:, :, i] for i in range(num_filters)]

        num_rows = num_filters // images_per_row

        fig = plt.figure()
        grid = ImageGrid(fig, 111, (num_rows, images_per_row))

        for ax, im in zip(grid, imgs):
            ax.imshow(im, cmap="viridis")

        plt.title(f"Convolutional Layer {idx + 1}")
        plt.show()


def visualize_conv_layers(model, imgs, conv_idx):
    """
    Function to visualize specified conv layers for given images.

    Args:
        model (tf.keras.Model): The model whose layers are to be visualized.
        imgs (Numpy array): The images for which the layers are to be visualized.
        conv_idx (list-like): The indices of the layers to be visualized (0-indexed).
    """
    activation_model = get_activation_model(model, conv_idx)
    activations = activation_model.predict(imgs)

    num_imgs = imgs.shape[0]
    num_layers = len(conv_idx)

    for idx in range(num_imgs):
        img_activs = [activations[i][idx, :, :, :] for i in range(num_layers)]
        _visualize_conv_layers_single_img(activations=img_activs, conv_idx=conv_idx)
