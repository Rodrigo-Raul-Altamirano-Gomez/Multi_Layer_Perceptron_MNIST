"""
Module that defines the architecture of the Multi-Layer Perceptron (MLP) neural network
from scratch using only NumPy.
"""

import numpy as np

class DenseNeuralNetwork:
    """
    Dense Neural Network with one hidden layer.
    """
    def __init__(self, input_dimension, hidden_dimension, output_dimension, learning_rate):
        """
        Initializes the weights and biases of the neural network.
        
        Args:
            input_dimension (int): Number of input features.
            hidden_dimension (int): Number of neurons in the hidden layer.
            output_dimension (int): Number of output neurons (classes).
            learning_rate (float): Learning rate for weight updates.
        """
        # He initialization for the weights of the first layer
        self.W1 = np.random.normal(scale=np.sqrt(2/input_dimension), size=(input_dimension, hidden_dimension))
        self.b1 = np.zeros(shape=(1, hidden_dimension))
        
        # Xavier/Glorot initialization for the output layer
        self.W2 = np.random.normal(scale=np.sqrt(1 / hidden_dimension), size=(hidden_dimension, output_dimension))
        self.b2 = np.zeros(shape=(1, output_dimension))
        
        self.learning_rate = learning_rate

    def _ReLU(self, Z):
        """
        Rectified Linear Unit (ReLU) activation function.
        """
        return np.maximum(0, Z)

    def _softmax(self, Z):
        """
        Softmax activation function to obtain probability distributions.
        Subtracts the maximum for numerical stability.
        """
        Z_normalized = Z - np.max(Z, axis=1, keepdims=True)
        Z_exp = np.exp(Z_normalized)
        return Z_exp / np.sum(Z_exp, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation.
        
        Args:
            X (numpy.ndarray): Input feature matrix.
            
        Returns:
            numpy.ndarray: Network predictions (probability distribution).
        """
        self.X = X
        
        # Hidden layer: Linear combination and ReLU activation
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._ReLU(self.Z1)
        
        # Output layer: Linear combination and Softmax activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._softmax(self.Z2)
        
        return self.A2

    def compute_loss(self, Y, A2):
        """
        Calculates the loss using Categorical Cross-Entropy.
        
        Args:
            Y (numpy.ndarray): True labels (one-hot encoding).
            A2 (numpy.ndarray): Network predictions.
            
        Returns:
            float: Average loss value of the batch.
        """
        # Add 1e-15 to prevent logarithm of zero
        return -1 * np.mean(np.sum(Y * np.log(A2 + 1e-15), axis=1))

    def backward(self, Y):
        """
        Backward propagation (Backpropagation) to calculate gradients and update weights.
        
        Args:
            Y (numpy.ndarray): True labels (one-hot encoding).
        """
        m = self.X.shape[0] # Batch size
        
        # Output layer gradients
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * np.dot(np.transpose(self.A1), dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dZ1 = np.dot(dZ2, np.transpose(self.W2)) * (self.Z1 > 0) # ReLU derivative is 1 if Z1 > 0, else 0
        dW1 = (1 / m) * np.dot(np.transpose(self.X), dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Weight and bias updates using Gradient Descent
        self.W1 = self.W1 - self.learning_rate*dW1
        self.b1 = self.b1 - self.learning_rate*db1
        self.W2 = self.W2 - self.learning_rate*dW2
        self.b2 = self.b2 - self.learning_rate*db2

    def predict(self, X):
        """
        Makes final predictions by returning the class with the highest probability.
        
        Args:
            X (numpy.ndarray): Data to predict.
            
        Returns:
            numpy.ndarray: Vector with predicted classes.
        """
        probability_distributions = self.forward(X)
        return np.argmax(probability_distributions, axis=1)

    def accuracy(self, prediction, Y):
        """
        Calculates the accuracy of the model.
        
        Args:
            prediction (numpy.ndarray): Predictions made by the model (indices).
            Y (numpy.ndarray): True labels in one-hot format.
            
        Returns:
            float: Accuracy percentage.
        """
        m = Y.shape[0]
        # Extract the true values corresponding to the predicted class
        Y_vs_prediction = Y[np.arange(m), prediction]
        total_true_predictions = np.sum(Y_vs_prediction)
        
        return (total_true_predictions / m) * 100