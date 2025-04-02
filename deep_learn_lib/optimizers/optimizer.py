from deep_learn_lib.nn import SequentialNet


class Optimizer:
    def step(self, net: SequentialNet):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def step(self, net: SequentialNet):
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
