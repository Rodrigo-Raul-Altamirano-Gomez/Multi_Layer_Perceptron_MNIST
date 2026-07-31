# MNIST / Fashion MNIST Classifier from Scratch

This project is a from-scratch implementation of a Dense Neural Network (Multi-Layer Perceptron or MLP) built using only NumPy in Python. Its main purpose is to classify images from the MNIST dataset (or variants like Fashion MNIST) without relying on external Deep Learning frameworks such as TensorFlow or PyTorch.

## Project Structure

The code is divided into three main files for better modularity and cleanliness:

- **`DataRetribution.py`**: Handles data downloading and preprocessing. It downloads the dataset via `OpenML`, normalizes the images so their pixel values are between 0 and 1, converts the labels to _one-hot_ format, and splits the data into training (80%) and testing (20%) sets.
- **`MLP.py`**: Defines the `DenseNeuralNetwork` class. It contains all the mathematical logic of the model, including:
  - Weight and bias initialization.
  - Activation functions (ReLU for the hidden layer and Softmax for the output).
  - Forward propagation (`forward`) and loss calculation (Categorical Cross-Entropy).
  - Backward propagation (`backward`) and weight updates via Gradient Descent.
  - Auxiliary functions for prediction and accuracy calculation.
- **`main.py`**: The main script that orchestrates the training. It defines the hyperparameters (input/output dimensions, learning rate, batch size, and epochs), trains the neural network using _mini-batches_, and prints the loss and accuracy throughout the process, ending with the test evaluation.

## Requirements

To run the code, you only need Python and a few standard libraries from the data science ecosystem:

- `numpy`
- `scikit-learn` (used solely to download the data from OpenML).

You can install them by running:
```bash
pip install numpy scikit-learn
```

## How to Run It

Simply execute the `main.py` script from your terminal:

```bash
python main.py
```

The script will automatically download the data (this may take a few seconds the first time), start the training loop for several epochs (15 by default), and show you the progress of the loss and accuracy in real-time. Upon completion, you will see the general accuracy calculated on the test data.

## About This Implementation

This project has a purely educational focus. It helps to practically understand "what happens behind the scenes" of a classic neural network training: from the linear algebra of inference, the chain rule for backpropagation, to how gradients and the learning rate progressively adjust the model's parameters.
