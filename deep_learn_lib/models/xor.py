import numpy as np

from deep_learn_lib.layers.layers import Linear, Tanh
from deep_learn_lib.nn import SequentialNet

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Xor
y = np.array([[0], [1], [1], [0]])

net = SequentialNet(
    [Linear(input_size=2, output_size=2), Tanh(), Linear(input_size=2, output_size=2)]
)
