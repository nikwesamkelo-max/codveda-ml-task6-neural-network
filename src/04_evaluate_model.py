"""Step 4 - Evaluate the trained network and visualize training vs validation loss."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import keras

DATA_PATH = "data/processed/mnist.npz"
MODEL_PATH = "artifacts/model_trained.keras"
HISTORY_PATH = "outputs/training_history.csv"
METRICS_PATH = "outputs/metrics.json"
CURVES_PATH = "images/training_curves.png"
MATRIX_PATH = "images/confusion_matrix.png"
WRONG_PATH = "images/misclassified_digits.png"


def save_figure(fig, path):
    os.makedirs("images", exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_curves(history):
    """Training vs validation loss (left) and accuracy (right), per epoch."""
    best_epoch = int(history["val_loss"].idxmin()) + 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, train_col, val_col, title in (
            (axes[0], "loss", "val_loss", "Loss (lower is better)"),
            (axes[1], "accuracy", "val_accuracy", "Accuracy (higher is better)")):
        ax.plot(history["epoch"], history[train_col], "o-", color="#2563eb", lw=2.5, label="Training")
        ax.plot(history["epoch"], history[val_col], "o-", color="#f97316", lw=2.5, label="Validation")
        ax.axvline(best_epoch, color="#16a34a", ls="--", lw=2)
        ax.set_title(title, fontsize=16, weight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=13)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].text(best_epoch + 0.1, axes[0].get_ylim()[1] * 0.9, f"best epoch: {best_epoch}",
                 color="#16a34a", fontsize=13, weight="bold")
    fig.suptitle("Training vs validation", fontsize=20, weight="bold")
    save_figure(fig, CURVES_PATH)


def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("Actual digit")
    ax.set_title("Predictions on 10,000 unseen test images", fontsize=17, weight="bold")
    save_figure(fig, MATRIX_PATH)


def plot_misclassified(x, y_true, y_pred, probs, n=10):
    """The images the network got wrong with the MOST confidence."""
    wrong = np.where(y_pred != y_true)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(axis=1))][:n]
    cols = 5
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows + 0.8), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), wrong):
        ax.imshow(x[idx], cmap="gray_r")
        ax.set_title(f"actual {y_true[idx]}\npredicted {y_pred[idx]} ({probs[idx].max():.0%})", fontsize=12)
    fig.suptitle("Where the network gets it wrong", fontsize=18, weight="bold")
    save_figure(fig, WRONG_PATH)


def per_digit_accuracy(matrix):
    return matrix.diagonal() / matrix.sum(axis=1)


def most_confused(matrix, top=3):
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:top]
    return [(int(a), int(p), int(off[a, p])) for a, p in order]


def main():
    data = np.load(DATA_PATH)
    x_test, y_test = data["x_test"], data["y_test"]
    model = keras.models.load_model(MODEL_PATH)
    history = pd.read_csv(HISTORY_PATH)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    probs = model.predict(x_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    matrix = confusion_matrix(y_test, y_pred, labels=range(10))
    digit_acc = per_digit_accuracy(matrix)
    best_epoch = int(history["val_loss"].idxmin()) + 1
    best = history.loc[best_epoch - 1]

    print(f"Test accuracy: {test_acc:.4f} ({int((y_pred == y_test).sum()):,} of {len(y_test):,} correct)")
    print(f"Test loss: {test_loss:.4f}\n")
    print(f"Accuracy at the best epoch ({best_epoch}):")
    print(f"  training {best['accuracy']:.4f} | validation {best['val_accuracy']:.4f} | test {test_acc:.4f}\n")
    print("Accuracy per digit:", {d: round(float(a), 3) for d, a in enumerate(digit_acc)})
    for actual, predicted, count in most_confused(matrix):
        print(f"Most confused: a {actual} read as {predicted}, {count} times")

    # Checks
    assert matrix.sum() == len(y_test)
    assert abs(test_acc - (y_pred == y_test).mean()) < 1e-3, "evaluate() and predict() disagree"
    assert test_acc > 0.95, "Test accuracy is unexpectedly low"
    print("\nEvaluation verified")

    os.makedirs("outputs", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"test_accuracy": float(test_acc), "test_loss": float(test_loss),
                   "best_epoch": best_epoch,
                   "train_accuracy_at_best_epoch": float(best["accuracy"]),
                   "validation_accuracy_at_best_epoch": float(best["val_accuracy"]),
                   "per_digit_accuracy": {str(d): float(a) for d, a in enumerate(digit_acc)}},
                  f, indent=2)
    plot_training_curves(history)
    plot_confusion_matrix(matrix)
    plot_misclassified(x_test, y_test, y_pred, probs)
    print(f"Saved -> {METRICS_PATH}, {CURVES_PATH}, {MATRIX_PATH}, {WRONG_PATH}")


if __name__ == "__main__":
    main()
