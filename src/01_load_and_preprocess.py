"""Step 1 - Load and preprocess the MNIST handwritten digits dataset."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras

OUT_PATH = "data/processed/mnist.npz"
CHART_PATH = "images/sample_digits.png"
VAL_SIZE = 10_000        # images held out to watch for overfitting while training
RANDOM_STATE = 42


def load_mnist():
    """Download MNIST (cached after the first run): 60,000 training and
    10,000 test images of handwritten digits, 28 x 28 pixels each."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


def scale_pixels(images):
    """Pixels are whole numbers 0-255. Scaling to 0-1 helps the network learn
    faster and more stably, because large inputs make training erratic."""
    return images.astype("float32") / 255.0


def make_validation_split(x, y):
    """Hold out part of the training set for validation. 'stratify' keeps every
    digit in the same proportion. The test set stays untouched until step 4."""
    return train_test_split(x, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE)


def plot_samples(x, y, per_digit=2):
    fig, axes = plt.subplots(per_digit, 10, figsize=(14, 1.6 * per_digit + 1))
    for digit in range(10):
        examples = np.where(y == digit)[0][:per_digit]
        for row, idx in enumerate(examples):
            ax = axes[row, digit]
            ax.imshow(x[idx], cmap="gray_r")
            ax.axis("off")
            if row == 0:
                ax.set_title(str(digit), fontsize=16, weight="bold")
    fig.suptitle("MNIST: handwritten digits, 28 x 28 pixels", fontsize=18, weight="bold")
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Training images: {x_train.shape} | Test images: {x_test.shape}")
    print(f"Pixel values before scaling: {x_train.min()} to {x_train.max()}")
    counts = {int(d): int(c) for d, c in zip(*np.unique(y_train, return_counts=True))}
    print("Images per digit (training):", counts, "\n")

    plot_samples(x_train, y_train)

    x_train, x_test = scale_pixels(x_train), scale_pixels(x_test)
    print(f"Pixel values after scaling: {x_train.min():.1f} to {x_train.max():.1f}")

    x_tr, x_val, y_tr, y_val = make_validation_split(x_train, y_train)
    print(f"Train: {len(x_tr):,} | Validation: {len(x_val):,} | Test: {len(x_test):,}\n")

    # Checks
    assert x_tr.shape[1:] == (28, 28) and x_test.shape[1:] == (28, 28)
    assert x_tr.dtype == np.float32 and 0.0 <= x_tr.min() and x_tr.max() <= 1.0
    assert set(np.unique(y_train)) == set(range(10)), "Expected digits 0-9"
    assert len(x_tr) + len(x_val) == 60_000
    share_tr = np.bincount(y_tr, minlength=10) / len(y_tr)
    share_val = np.bincount(y_val, minlength=10) / len(y_val)
    assert np.abs(share_tr - share_val).max() < 0.005, "Validation split is not balanced"
    print("Preprocessing verified")

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(OUT_PATH, x_train=x_tr, y_train=y_tr, x_val=x_val, y_val=y_val,
                        x_test=x_test, y_test=y_test)
    print(f"Saved -> {OUT_PATH} and {CHART_PATH}")


if __name__ == "__main__":
    main()
