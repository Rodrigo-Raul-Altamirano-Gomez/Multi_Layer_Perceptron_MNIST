"""
Main script for training and evaluating the neural network.
Loads data, trains the model in mini-batches, and calculates accuracy.
"""

from DataRetribution import get_data
import MLP
import numpy as np

# Network hyperparameters and layer dimensions
input_dimension, hidden_dimension, output_dimension = 784, 128, 10
learning_rate, batch_size, epochs = 0.05, 128, 15

# Initialization of the Dense Neural Network (Multilayer Perceptron)
multi_layer_perceptron = MLP.DenseNeuralNetwork(input_dimension, hidden_dimension, output_dimension, learning_rate)

# Obtain the training and testing datasets
X_train, X_test, Y_train, Y_test = get_data()
m_train = X_train.shape[0]

# Main training loop
for epoch in range(epochs):
    # Shuffle the data at the beginning of each epoch for better generalization
    permuted_indexes = np.random.permutation(m_train)
    X_train_shuffled = X_train[permuted_indexes]
    Y_train_shuffled =  Y_train[permuted_indexes]
    
    # Train by iterating over the mini-batches
    for batch in range((m_train + batch_size - 1) // batch_size):
        # Determine the start and end indices of the current batch
        X_train_batch = X_train_shuffled[batch * batch_size : min(((batch + 1) * batch_size), m_train)]
        Y_train_batch = Y_train_shuffled[batch * batch_size : min(((batch + 1) * batch_size), m_train)]
        
        # Forward propagation
        predicted_distributions = multi_layer_perceptron.forward(X_train_batch)
        
        # Backward propagation and weight update
        multi_layer_perceptron.backward(Y_train_batch)
        
    # Evaluate at the end of the epoch using the entire training set
    complete_distributions = multi_layer_perceptron.forward(X_train)
    total_loss = multi_layer_perceptron.compute_loss(Y_train, complete_distributions)
    complete_predictions = multi_layer_perceptron.predict(X_train)
    accuracy = multi_layer_perceptron.accuracy(complete_predictions, Y_train)
    
    # Display progress
    print(f"Epoch: {epoch}/{epochs} - Loss: {total_loss} - Accuracy: {accuracy}%")

# Final evaluation with the test set that the model has never seen
test_predictions = multi_layer_perceptron.predict(X_test)
general_accuracy = multi_layer_perceptron.accuracy(test_predictions, Y_test)

print("General accuracy: ", general_accuracy)