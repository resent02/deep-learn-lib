from deep_learn_lib.nn import SequentialNet
from deep_learn_lib.optimizers.sgd import Optimizer

import numpy as np

class SGD_momentum(Optimizer):
    """
    Stochastic Gradient Descent with momentum:
    v_t+1 = mc * vt + lr * grad
    param_t+1 = param - v_t+1
    """

    def __init__(self, lr: float = 0.001, mc: float = 0.9) -> None:
        self.lr = lr
        self.mc = mc

    def step(self, net: SequentialNet) -> None:
        for param, grad, velocity in net.params_and_velocities():
            velocity = self.mc * velocity + self.lr * grad
            param -= velocity

