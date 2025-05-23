# Deep Learn Lib

A lightweight deep learning library built from scratch with NumPy. Implements layers, loss functions, and optimizers for educational and research purposes.

## Installation

### Prerequisites
- Python 3.11+
- NumPy

### Option 1: Install from source (recommended for development)
```bash
git clone 
cd deep_learn_lib
pip install -e .  # Editable install
```



## Quick Start

```python
import numpy as np

from deep_learn_lib.layers import Linear, Tanh
from deep_learn_lib.losses import MSE
from deep_learn_lib.nn import SequentialNet
from deep_learn_lib.optimizers import SGD

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
optimizer = SGD(lr=0.01)

for epoch in range(100):
    # Forward pass
    output = model.forward(X)

    # Compute loss
    loss = mse.loss(output, y)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Backward pass
    grad = mse.grad(output, y)
    model.backward(grad)

    # Update weights
    optimizer.step(model)

```
