from deep_learn_lib.nn import SequentialNet
from deep_learn_lib.optimizers.sgd import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer
    
    """

    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def step(self, net: SequentialNet) -> float:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
