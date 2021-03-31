import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from data import get_test_data_generator
from model import predict


def plot_loss_curve(history):
    plt.plot(history.history['loss'], 'r', label='train')
    plt.plot(history.history['val_loss'], 'g', label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def get_classification_report(
    model, data_dir, batch_size=64,
    steps=None, threshold=0.5, output_dict=False
):
    data = get_test_data_generator(data_dir, batch_size=batch_size)
    predictions = predict(model, data, steps, threshold)
    predictions = predictions.reshape((predictions.shape[0],))
    return classification_report(data.classes, predictions, output_dict=output_dict)


def visualize_hidden_layers(model, ip):
    pass
