"""Step 2 - Design the neural network architecture."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tensorflow import keras

MODEL_PATH = "artifacts/model_untrained.keras"
SUMMARY_PATH = "outputs/model_summary.txt"
CHART_PATH = "images/architecture.png"

INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10          # digits 0-9
HIDDEN_UNITS = (128, 64)  # neurons in the two hidden layers
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
RANDOM_STATE = 42


def build_model():
    """Feed-forward network: input -> two hidden layers -> output."""
    keras.utils.set_random_seed(RANDOM_STATE)
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),                                # 28 x 28 pixel image
        keras.layers.Flatten(),                                        # grid -> 784 numbers
        keras.layers.Dense(HIDDEN_UNITS[0], activation="relu"),        # hidden layer 1
        keras.layers.Dropout(DROPOUT_RATE),                            # randomly switch off 20% while training
        keras.layers.Dense(HIDDEN_UNITS[1], activation="relu"),        # hidden layer 2
        keras.layers.Dense(NUM_CLASSES, activation="softmax"),         # one probability per digit
    ], name="mnist_feed_forward")
    # Adam adjusts the weights during backpropagation; the loss compares the
    # predicted probabilities with the true digit (labels are whole numbers 0-9).
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def expected_param_count():
    """Each Dense layer has (inputs x neurons) weights + one bias per neuron."""
    sizes = [int(np.prod(INPUT_SHAPE)), *HIDDEN_UNITS, NUM_CLASSES]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def describe_layers(model):
    """Turn the model's layers into plain descriptions for the diagram."""
    dense = [layer for layer in model.layers if type(layer).__name__ == "Dense"]
    specs = [{"kind": "Input", "title": "Input",
              "detail": f"{INPUT_SHAPE[0]} x {INPUT_SHAPE[1]}\nimage", "params": 0}]
    hidden_number = 0
    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "Flatten":
            specs.append({"kind": "Flatten", "title": "Flatten",
                          "detail": f"{int(np.prod(INPUT_SHAPE))}\nvalues", "params": 0})
        elif kind == "Dense":
            is_output = layer is dense[-1]
            if not is_output:
                hidden_number += 1
            specs.append({"kind": "Output" if is_output else "Hidden",
                          "title": "Output layer" if is_output else f"Hidden layer {hidden_number}",
                          "detail": f"{layer.units} neurons\n{layer.activation.__name__}",
                          "params": layer.count_params()})
        elif kind == "Dropout":
            specs.append({"kind": "Dropout", "title": "Dropout",
                          "detail": f"{layer.rate:.0%} off\nwhile training", "params": 0})
    return specs


def plot_architecture(specs, total_params):
    colors = {"Input": "#9ca3af", "Flatten": "#9ca3af", "Hidden": "#2563eb",
              "Dropout": "#f97316", "Output": "#16a34a"}
    n = len(specs)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, n * 2.4); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(n * 1.2, 4.6, "Neural network architecture", ha="center", fontsize=22, weight="bold")
    ax.text(n * 1.2, 4.05, f"{total_params:,} trainable parameters", ha="center",
            fontsize=14, color="#6b7280")
    for i, s in enumerate(specs):
        x = 0.2 + i * 2.4
        ax.add_patch(FancyBboxPatch((x, 0.6), 1.9, 2.6, boxstyle="round,pad=0.05",
                                    fc=colors[s["kind"]], ec="none"))
        ax.text(x + 0.95, 2.8, s["title"], ha="center", color="white", fontsize=13, weight="bold")
        ax.text(x + 0.95, 1.95, s["detail"], ha="center", va="center", color="white", fontsize=13)
        if s["params"]:
            ax.text(x + 0.95, 0.95, f"{s['params']:,}\nparams", ha="center", va="center",
                    color="white", fontsize=11)
        if i < n - 1:
            ax.annotate("", xy=(x + 2.35, 1.9), xytext=(x + 1.97, 1.9),
                        arrowprops=dict(arrowstyle="->", color="#6b7280", lw=2.2))
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    model = build_model()

    lines = []
    model.summary(print_fn=lambda line, **kwargs: lines.append(line))
    print("\n".join(lines), "\n")

    # Checks
    assert model.count_params() == expected_param_count(), "Parameter count does not match the design"
    dense_layers = [l for l in model.layers if type(l).__name__ == "Dense"]
    assert len(dense_layers) == 3 and dense_layers[-1].units == NUM_CLASSES
    assert dense_layers[-1].activation.__name__ == "softmax"
    sample = np.random.default_rng(0).random((5, *INPUT_SHAPE), dtype=np.float32)
    probabilities = model.predict(sample, verbose=0)
    assert probabilities.shape == (5, NUM_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5), "Outputs should be probabilities"
    print(f"Architecture verified: {model.count_params():,} parameters; "
          "an untrained network outputs valid probabilities")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    model.save(MODEL_PATH)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    plot_architecture(describe_layers(model), model.count_params())
    print(f"Saved -> {MODEL_PATH}, {SUMMARY_PATH}, {CHART_PATH}")


if __name__ == "__main__":
    main()
