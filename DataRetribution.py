"""
Module for downloading and preprocessing data.
Uses the 'mnist_784' dataset from OpenML.
"""

import sklearn.datasets as dt
import numpy as np

def download_data(dataset):
    """
    Downloads the specified dataset from OpenML.
    
    Args:
        dataset (str): Name of the dataset to download.
        
    Returns:
        tuple: (X, y) where X are the features (images) and y are the labels.
    """
    # fetch_openml gets the dataset, as_frame=False ensures it returns numpy arrays
    data = dt.fetch_openml(dataset, as_frame=False)
    X = data["data"]
    y = data["target"]
    return (X, y)

def transform_data(X, y):
    """
    Normalizes the features and converts labels to one-hot format.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Label vector.
        
    Returns:
        tuple: (X_normalized, Y_onehot) preprocessed data ready for the network.
    """
    # Normalize pixel values (from 0-255 to 0.0-1.0)
    X_normalized = X / 255
    
    # Ensure labels are integers
    y_int = y.astype(int)
    m = len(y)
    
    # Create the one-hot encoding matrix filled with zeros
    Y_onehot = np.zeros(shape=(m, 10))
    
    # Assign 1 in the corresponding position for each label
    Y_onehot[np.arange(m), y_int] = 1
    
    return (X_normalized, Y_onehot)

def slice_data(X, Y, test_proportion):
    """
    Splits the dataset into training and testing sets.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        Y (numpy.ndarray): One-hot encoded labels matrix.
        test_proportion (float): Proportion of the dataset to use for testing (e.g., 0.2 for 20%).
        
    Returns:
        tuple: (X_train, X_test, Y_train, Y_test)
    """
    # Calculate the split index based on the proportion
    split_index = int((1 - test_proportion) * X.shape[0])
    
    # Split the features
    X_train = X[:split_index]
    X_test = X[split_index:]
    
    # Split the labels
    Y_train = Y[:split_index]
    Y_test = Y[split_index:]
    
    return (X_train, X_test, Y_train, Y_test)

def get_data():
    """
    Main function to fetch, process, and split the data.
    
    Returns:
        tuple: Training and testing sets (X_train, X_test, Y_train, Y_test).
    """
    # Download MNIST data (784 features = 28x28 pixels)
    X, y = download_data("mnist_784")
    
    # Transform and normalize the data
    X, Y = transform_data(X, y)
    
    # Split using the specified test proportion (20% test, 80% train)
    return slice_data(X, Y, 0.2)