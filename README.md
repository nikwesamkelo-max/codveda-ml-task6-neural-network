Neural Network with TensorFlow/Keras: Handwritten Digit Classification

Codveda Technology — Machine Learning Internship | Task 6

A reproducible machine learning pipeline that builds, trains, and evaluates a feed-forward neural network to classify handwritten digits (0–9) using the MNIST dataset.

The model achieved 97.88% test accuracy on 10,000 previously unseen images.

Project Overview

This project demonstrates a complete machine learning workflow, from dataset preparation and model construction to training, evaluation, and error analysis.

Dataset

MNIST contains 70,000 grayscale images of handwritten digits. Each image is 28 × 28 pixels.

Dataset| Images| Purpose
Training| 50,000| Learning model weights
Validation| 10,000| Monitoring generalization and overfitting
Test| 10,000| Final evaluation on unseen data
Total| 70,000| 

The original 60,000-image MNIST training set was divided into 50,000 training images and 10,000 validation images. The original 10,000-image test set remained separate until final evaluation.

Sample Images

"MNIST handwritten digit samples" (images/sample_digits.png)

Machine Learning Pipeline

Step| Script| Description
1| "01_load_and_preprocess.py"| Loads MNIST, scales pixel values to 0–1, and creates a stratified validation split
2| "02_build_model.py"| Builds and compiles the neural network and generates an architecture diagram
3| "03_train_model.py"| Trains the model using backpropagation and early stopping
4| "04_evaluate_model.py"| Evaluates test performance and generates training curves, a confusion matrix, and misclassification visualizations

Model Architecture

The network uses a flattened image representation followed by two fully connected hidden layers and a softmax output layer.

Layer| Configuration| Parameters
Input| 28 × 28 grayscale image| 0
Flatten| Converts the image into 784 features| 0
Dense| 128 neurons, ReLU activation| 100,480
Dropout| 20% dropout rate| 0
Dense| 64 neurons, ReLU activation| 8,256
Output| 10 neurons, softmax activation| 650
Total| 109,386 trainable parameters| 109,386

Architecture Diagram

"Neural network architecture" (images/architecture.png)

Key Design Decisions

- Pixel normalization: Converts pixel values from 0–255 to 0–1 to provide consistently scaled inputs.
- Stratified validation split: Reserves 10,000 images for validation while preserving the distribution of digit classes.
- ReLU activation: Introduces non-linearity into the hidden layers, allowing the network to learn more complex patterns.
- Softmax output: Produces probabilities across the ten possible digit classes.
- Dropout: Randomly disables 20% of the relevant hidden-layer activations during training to help reduce overfitting.
- Adam optimizer: Updates model weights during training using gradient-based optimization.
- Sparse categorical cross-entropy: Calculates classification loss directly from integer labels, without requiring one-hot encoding.
- Early stopping: Stops training when validation loss fails to improve for three consecutive epochs and restores the best model weights.
- Reproducibility: Uses a fixed random state of 42 for the stratified train-validation split.

Training Configuration

Setting| Value
Optimizer| Adam
Loss function| Sparse categorical cross-entropy
Batch size| 128
Maximum epochs| 20
Early stopping patience| 3 epochs
Dropout rate| 20%
Random state for data split| 42

Results

The trained model correctly classified 9,788 of 10,000 test images, achieving 97.88% test accuracy.

Overall Performance

Metric| Result
Training accuracy at best epoch| 98.65%
Validation accuracy at best epoch| 97.89%
Test accuracy| 97.88%
Test loss| 0.0800
Best epoch| 13
Total epochs completed| 16
Correct test predictions| 9,788
Incorrect test predictions| 212
Trainable parameters| 109,386

Training and Validation Curves

"Training and validation accuracy and loss" (images/training_curves.png)

The model's training and validation accuracy generally increased while their losses decreased. Validation loss reached its minimum at epoch 13, at 0.0826. Training continued until epoch 16, when early stopping terminated the run and restored the best weights.

The small difference between training accuracy (98.65%) and validation accuracy (97.89%) at the best epoch suggests that the model generalizes well, although some overfitting remains.

Error Analysis

Despite its high overall accuracy, the model occasionally confused visually similar handwritten digits.

Most Frequent Misclassifications

Actual Digit| Predicted Digit| Errors
5| 3| 17
9| 4| 12
7| 2| 9

Accuracy by Digit

Digit| Accuracy
0| 99.0%
1| 98.9%
2| 97.9%
3| 98.4%
4| 98.6%
5| 96.1%
6| 98.3%
7| 97.8%
8| 96.9%
9| 96.7%

Digit 5 had the lowest reported classification accuracy at 96.1%. Differences in handwriting styles may explain some errors, particularly when a digit resembles another class.

Misclassified Examples

The visualization below displays test images that the model classified incorrectly with high confidence.

"Misclassified handwritten digits" (images/misclassified_digits.png)

Confusion Matrix

The confusion matrix shows how frequently each actual digit was classified as each predicted digit, helping identify patterns in the model's errors.

"Confusion matrix" (images/confusion_matrix.png)

Limitations and Future Improvements

Limitations

- Limited spatial feature learning: Flattening each image loses the explicit spatial structure of neighboring pixels. A convolutional neural network (CNN) could learn local image features more effectively.
- Single training run: The reported results come from one experiment. Additional runs with different random seeds would help establish performance consistency.
- Ambiguous handwriting: Similar-looking digits remain a source of classification errors.
- Limited dataset scope: MNIST contains small grayscale images of isolated digits, so performance may differ on real-world images with different backgrounds, image quality, or writing styles.

Potential Improvements

- Build and evaluate a CNN architecture.
- Compare alternative network architectures and dropout rates.
- Examine high-confidence misclassifications to identify recurring patterns.
- Repeat experiments with different random seeds.
- Evaluate model performance using precision, recall, and F1-score alongside accuracy.

Installation and Execution

Requirements

Python and the dependencies listed in "requirements.txt" are required.

Install dependencies:

pip install -r requirements.txt

Run the complete pipeline:

python run_all.py

The pipeline downloads MNIST if necessary, preprocesses the data, builds the model, trains it, and evaluates its performance.

Run in Google Colab

Open "Google Colab" (https://colab.new), create a new notebook, and run the following commands in a code cell:

!git clone https://github.com/nikwesamkelo-max/codveda-ml-task6-neural-network.git
%cd codveda-ml-task6-neural-network
!pip install -q -r requirements.txt
!python run_all.py

If the repository is already cloned in the current Colab session, navigate to its existing directory instead of cloning it again.

Note: The first run requires an internet connection to download the MNIST dataset. Training time depends on the available runtime.

Generated Outputs

The pipeline generates the following artifacts:

Location| Contents
"data/processed/"| Preprocessed MNIST dataset
"artifacts/"| Saved model files
"outputs/"| Metrics, model summary, and training history
"images/"| Sample digits, architecture, training curves, confusion matrix, and misclassified examples

Generated files may be excluded from Git using ".gitignore", depending on the repository configuration.

Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- Git and GitHub

Author

Samkelo

- GitHub: "nikwesamkelo-max" (https://github.com/nikwesamkelo-max)
- LinkedIn: "Samkelo on LinkedIn" (https://linkedin.com/in/samkelo-nikwe-93226b344)

---

Developed as part of the Codveda Technology Machine Learning Internship.
