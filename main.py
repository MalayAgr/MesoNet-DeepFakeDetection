from model import build_model, save_model_history
from train import train_model


def main():
    train_data_dir = 'data/train/'

    val_split, epochs, batch_size = 0.20, 50, 64

    decay_rate, decay_limit = 0.10, 1e-6
    history_filename = 'model_history.pckl'

    model = build_model()

    history = train_model(
        model,
        train_data_dir,
        batch_size=batch_size,
        validation_split=val_split,
        epochs=epochs,
        decay_rate=decay_rate,
        decay_limit=decay_limit,
    )

    save_model_history(history, history_filename)


if __name__ == '__main__':
    main()
