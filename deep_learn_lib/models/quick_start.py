import numpy as np

from deep_learn_lib.layers import Linear, Tanh
from deep_learn_lib.losses import MSE
from deep_learn_lib.nn import SequentialNet
from deep_learn_lib.optimizers import SGD
from deep_learn_lib.optimizers import SGD_momentum

# Create a neural network
model = SequentialNet(
    [
        Linear(input_size=2, output_size=10),
        Tanh(),
        Linear(input_size=10, output_size=1),
    ]
)

# Sample data
X = np.random.randn(100, 2)
y = np.random.randn(100, 1)

# Training loop
mse = MSE()
optimizer = SGD_momentum(lr=0.01)

for epoch in range(10000000000000000000000):
    # Forward pass
    output = model.forward(X)

    # Compute loss
    loss = mse.loss(output, y)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Backward pass
    grad = mse.grad(output, y)
    model.backward(grad)
    if loss < 0.6: 
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        break

    # Update weights
    optimizer.step(model)
