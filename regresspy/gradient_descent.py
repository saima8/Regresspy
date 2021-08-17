from typing import Dict, Tuple

import numpy as np
from numpy import ndarray


def forward(X: ndarray, Y: ndarray, weights: Dict[str, ndarray]) ->  Tuple[float, Dict[str, ndarray]]:
    """Performs a single forward pass of the gradient descent.

    Args:
        X (ndarray): the input data.
        Y (ndarray): the ground truths.
        weights (Dict[str, ndarray]): a dictionary containing weights and bias.
    Returns:
        (Tuple[float, Dict[str, ndarray]]): prediction, dictionary of weights and bias
    """
    w = weights['W']
    b = weights['B']

    # Matrix multiplication of X and weights.
    # Caution: The matrix shapes of X and weights have to be concerned 
    N = np.dot(X,w)
    # Bias addition with broadcasting and calculated Prediction as P
    P = N + b
    # Calculated loss function using Mean Squared Error
    loss = np.mean(np.power((Y-P), 2))


    # Intermediate values till Loss function are stored and returned in a Dictionary
    info = {}
    info['X'] = X
    info['Y'] = Y
    info['N'] = N
    info['P'] = P

    return loss, info


def backward(info: Dict[str, ndarray], weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """Computes and returns the gradients of weights and bias.

    Args:
        info (Dict[str, ndarray]): dictionary containing X, Y, N and P.
        weights (Dict[str, ndarray]): dictionary of weights and bias.
    Returns:
        (Dict[str, ndarray]): Dictionary of gradients of weights and bias.
    """
    # Derivative of the MSE at its input P
    dLdP = -2 * (info['Y'] - info['P'])
    # Partial derivatives of the bias addition operation
    dPdN = np.ones_like(info['N'])
    dPdB = np.ones_like(weights['B'])
    # Derivative of the MSE with respect to the outcome of matrix multiplication
    dLdN = dLdP * dPdN
    # Partial derivative of matrix multiplication with respect to weights
    dNdW = np.transpose(info['X'], (1, 0))
    # Derivatives of MSE with respect to weights
    dLdW = np.dot(dNdW, dLdN)
    # Derivatives of MSE with respect to bias
    dLdB = (dLdP * dPdB).sum(axis=0)

    gradients = {}
    gradients['W'] = dLdW
    gradients['B'] = dLdB
    
    return gradients
