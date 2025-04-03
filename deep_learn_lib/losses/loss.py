"""
Loss function measures how good predictions
"""

import numpy as np

from deep_learn_lib.utils.tensor import Tensor


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    Mean Squared error
    mse = 1/n * sum[(predicted-actual)^2]
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return float(np.mean((predicted - actual) ** 2))

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual) / np.size(predicted)  # Added normalization
