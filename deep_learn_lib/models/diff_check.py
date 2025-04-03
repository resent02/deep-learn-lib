import numpy as np
import torch
import torch.nn as nn

from deep_learn_lib.layers import Linear, Tanh
from deep_learn_lib.losses import MSE
from deep_learn_lib.nn import SequentialNet
from deep_learn_lib.optimizers import SGD

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Generate synthetic data
def generate_data(n_samples=1000):
    X = np.random.randn(n_samples, 3)
    y = np.sin(X[:, [0]]) + np.cos(X[:, [1]]) + X[:, [2]] * 0.5
    return X, y


X, y = generate_data()
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)

# Create your custom network
custom_net = SequentialNet(
    [
        Linear(input_size=3, output_size=10),
        Tanh(),
        Linear(input_size=10, output_size=5),
        Tanh(),
        Linear(input_size=5, output_size=1),
    ]
)


# Initialize with specific weights for comparison
def init_weights(layer, torch_layer=None):
    if isinstance(layer, Linear):
        if torch_layer is not None:
            # Copy PyTorch's initialization to your network
            layer.params["w"] = torch_layer.weight.data.numpy().T
            layer.params["b"] = torch_layer.bias.data.numpy()
        else:
            # Use fixed initialization for comparison
            layer.params["w"] = np.random.randn(*layer.params["w"].shape) * 0.1
            layer.params["b"] = np.random.randn(*layer.params["b"].shape) * 0.1


# Create equivalent PyTorch network
torch_net = nn.Sequential(
    nn.Linear(3, 10), nn.Tanh(), nn.Linear(10, 5), nn.Tanh(), nn.Linear(5, 1)
)

# Initialize both networks with the same weights
for custom_layer, torch_layer in zip(custom_net.layers, torch_net):
    if isinstance(custom_layer, Linear):
        init_weights(custom_layer, torch_layer)

# Training setup
custom_loss_fn = MSE()
custom_optimizer = SGD(lr=0.01)

torch_loss_fn = nn.MSELoss()
torch_optimizer = torch.optim.SGD(torch_net.parameters(), lr=0.01)

# Comparison test
print("Epoch | Custom Loss | Torch Loss | Output Diff | Grad Diff")
print("--------------------------------------------------------")

for epoch in range(100):
    # Forward pass comparison
    custom_output = custom_net.forward(X)
    torch_output = torch_net(X_torch).detach().numpy()

    # Loss comparison
    custom_loss = custom_loss_fn.loss(custom_output, y)
    torch_loss = torch_loss_fn(torch_net(X_torch), y_torch).item()

    # Backward pass (custom)
    grad = custom_loss_fn.grad(custom_output, y)
    custom_net.backward(grad)
    custom_optimizer.step(custom_net)

    # Backward pass (torch)
    torch_optimizer.zero_grad()
    torch_loss_fn(torch_net(X_torch), y_torch).backward()
    torch_optimizer.step()

    # Calculate differences
    output_diff = np.mean(np.abs(custom_output - torch_output))

    # Compare gradients (for first layer)
    custom_grad = custom_net.layers[0].grads["w"]
    torch_grad = torch_net[0].weight.grad.numpy().T
    grad_diff = np.mean(np.abs(custom_grad - torch_grad))

    if epoch % 10 == 0:
        print(
            f"{epoch:5d} | {custom_loss:.6f} | {torch_loss:.6f} | {output_diff:.6f} | {grad_diff:.6f}"
        )

# Final weight comparison
print("\nWeight comparison for first layer:")
print("Custom weights:\n", custom_net.layers[0].params["w"][:3, :3])
print("Torch weights:\n", torch_net[0].weight.data.numpy()[:3, :3].T)
