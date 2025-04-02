import numpy as np

from src.layers import Linear, Tanh
from src.nn import SequentialNet

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Xor
y = np.array([[0], [1], [1], [0]])

net = SequentialNet(
    [Linear(input_size=2, output_size=2), Tanh(), Linear(input_size=2, output_size=2)]
)

for _ in range(100):
    net.forward()
