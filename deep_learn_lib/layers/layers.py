from typing import Callable, Dict

import numpy as np

from deep_learn_lib.utils.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.velocity : Dict[str,Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produces the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropogate the gradient through layers
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes output = inputs @ w + b"""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)
        self.velocity["w"] = np.zeros((input_size, output_size))
        self.velocity["b"] = np.zeros(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """

        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a*b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """

        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def sigm(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-x))


def sigm_prime(x: Tensor) -> Tensor:
    return sigm(x) * (1 - sigm(x))


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigm, sigm_prime)
