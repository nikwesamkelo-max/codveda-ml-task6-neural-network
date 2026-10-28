"""Run the full neural network pipeline (steps 1-4) in order."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    "src/01_load_and_preprocess.py",
    "src/02_build_model.py",
    "src/03_train_model.py",
    "src/04_evaluate_model.py",
]


def main():
    print("Note: step 1 downloads MNIST on the first run (internet needed) and "
          "step 3 trains the network (a few minutes).")
    total_start = time.time()
    for number, step in enumerate(STEPS, start=1):
        print(f"\n{'=' * 60}\nSTEP {number}/{len(STEPS)}: {step}\n{'=' * 60}")
        start = time.time()
        result = subprocess.run([sys.executable, step], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"\nStep failed: {step}")
        print(f"Finished in {time.time() - start:.1f}s")

    print(f"\nPipeline complete in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
