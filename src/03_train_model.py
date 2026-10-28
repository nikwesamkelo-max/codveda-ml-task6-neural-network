"""Step 3 - Train the neural network with backpropagation."""
import os
import numpy as np
import pandas as pd
from tensorflow import keras

DATA_PATH = "data/processed/mnist.npz"
MODEL_IN = "artifacts/model_untrained.keras"
MODEL_OUT = "artifacts/model_trained.keras"
HISTORY_PATH = "outputs/training_history.csv"

EPOCHS = 20        # maximum passes over the 50,000 training images
BATCH_SIZE = 128   # images processed before each weight update
PATIENCE = 3       # stop if validation loss has not improved for 3 epochs
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    data = np.load(path)
    return data["x_train"], data["y_train"], data["x_val"], data["y_val"]


def train(model, x_train, y_train, x_val, y_val):
    """model.fit runs backpropagation. For each batch of images it (1) makes
    predictions, (2) measures the error with the loss function, (3) sends that
    error backwards through the layers to work out how each weight contributed,
    and (4) lets the Adam optimizer nudge the weights to reduce the error.

    EarlyStopping watches the validation loss (images the network is NOT trained
    on) and stops when it stops improving, keeping the best weights. This
    prevents the network from memorizing the training images."""
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    return model.fit(x_train, y_train, validation_data=(x_val, y_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=[early_stop], verbose=2)


def summarize_history(history):
    """Turn Keras' history dictionary into a table; find the best epoch."""
    table = pd.DataFrame(history)
    table.insert(0, "epoch", range(1, len(table) + 1))
    best_epoch = int(table["val_loss"].idxmin()) + 1
    return table, best_epoch


def main():
    x_train, y_train, x_val, y_val = load_data()
    model = keras.models.load_model(MODEL_IN)
    print(f"Training on {len(x_train):,} images, validating on {len(x_val):,}\n")

    keras.utils.set_random_seed(RANDOM_STATE)
    history = train(model, x_train, y_train, x_val, y_val)

    table, best_epoch = summarize_history(history.history)
    best = table.loc[best_epoch - 1]
    print(f"\nTrained for {len(table)} epochs; best epoch: {best_epoch}")
    print(f"At the best epoch: training accuracy {best['accuracy']:.4f}, "
          f"validation accuracy {best['val_accuracy']:.4f}")

    # Checks
    assert table["loss"].iloc[-1] < table["loss"].iloc[0], "Training loss did not fall"
    assert best["val_accuracy"] > 0.95, "Validation accuracy is unexpectedly low"
    print("Training verified. Test images untouched until step 4")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    model.save(MODEL_OUT)
    table.round(5).to_csv(HISTORY_PATH, index=False)
    print(f"Saved -> {MODEL_OUT} and {HISTORY_PATH}")


if __name__ == "__main__":
    main()
