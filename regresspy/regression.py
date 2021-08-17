from typing import Dict, Tuple

import numpy as np
from numpy import ndarray

from typing import Dict, Tuple

import numpy as np
from numpy import ndarray

from regresspy import loss
from loss import mae, sse, mse, rmse
from gradient_descent import forward, backward


class Regression(object):
    def __init__(self, epochs=50, learning_rate=0.001):
        self._epochs = epochs
        self._lr = learning_rate
        self._weights = {}
        self._X = None
        self._Y = None
    
    def fit(self, X: ndarray, Y:ndarray) -> None:
        """Fits the data using gradient descent algorithm.

        Args:
            X (ndarray): Should be of shape (observations x features)
            Y (ndarray): Should be of shape (observations x 1)
        """
        assert X.shape[0] == Y.shape[0]

        self._initialize_weights(X.shape)
        assert self._weights['W'].shape == (X.shape[1], 1)
        assert self._weights['B'].shape == (1, 1)

        self._train(X, Y)

    def predict(self, X: ndarray) -> ndarray:
        """Predicts new data with the fitted weights and bias.

        Args:
            X (ndarray): new dataset of shape (observations x features)
        Returns
            (ndarray): predictions of shape (observations x 1)
        """
        predictions = X @ self._weights['W'] + self ._weights['B'] 
        return predictions

    def score(self, X: ndarray, Y: ndarray, metric='rmse') -> float:
        """Returns the score of the fitted data. Possible metrics include
        'mae', 'sse', 'mse', and 'rmse'.

        Args:
            X (ndarray): dataset of shape (observations x features)
            Y (ndarray): labels of shape (observations x 1)
            metric (str): 'mae', 'sse', 'mse', or 'rmse'
        Returns:
            (float): score of the fitted line.
        """
        metrics = {
            'mae': mae,
            'sse': sse,
            'mse': mse,
            'rmse': rmse
        }

        # predictions = X*self._weights['W'] + self._weights['B']
        predictions = _X @ self._weights['W'] + self ._weights['B']
        score = metric(predictions, Y) #TODO
        return score

    def _initialize_weights(self, shape: Tuple[int, int]) -> None:
        """The shape of the weights will be (features x 1), and the shape
        of the bias will be (1,1).
        """
        self._weights = {
            'W': np.random.rand(self._X.shape[1],1), 
            'B': np.random.rand(1,1)   
        }
    
    def _train(self, X: ndarray, Y: ndarray) -> None:
        """Train data using gradient descent
        """
        for i in range(self._epochs):
            print('Epoch: ', i+1)
            #Computes forward propagation 
            loss, info = forward(X, Y, self._weights) 
            print('Loss: ', loss)
            #Computes backward propagation
            grads = backward(info, self._weights) 
            self._weights['W'] = self._weights['W'] - self._lr * [grads['W']]  
            self._weights['B'] = self._weights['B'] - self._lr * [grads['B']] 