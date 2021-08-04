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
    w = weights['w']
    b = weights['b']

    N = #TODO Matrix Multiplication -> np.dot(X,weights)
    P = #TODO Adding Bias -> N + b
    loss = #TODO Compute Mean Squared Error -> np.mean(np.power((Y-P), 2))

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
