import numpy as np

from deep_learn_lib.layers import Linear, Sigmoid, Tanh
from deep_learn_lib.losses import MSE
from deep_learn_lib.nn import SequentialNet
from deep_learn_lib.optimizers import SGD


# Layer Tests
def test_linear_layer_initialization():
    input_size, output_size = 10, 5
    linear = Linear(input_size, output_size)

    assert linear.params["w"].shape == (input_size, output_size)
    assert linear.params["b"].shape == (output_size,)


def test_linear_layer_forward():
    input_size, output_size = 3, 2
    linear = Linear(input_size, output_size)

    # Set fixed weights and bias for predictable testing
    linear.params["w"] = np.array([[1, 2], [3, 4], [5, 6]])
    linear.params["b"] = np.array([0.5, 1.5])

    inputs = np.array([1, 2, 3])
    expected_output = np.array(
        [1 * 1 + 2 * 3 + 3 * 5 + 0.5, 1 * 2 + 2 * 4 + 3 * 6 + 1.5]
    )

    output = linear.forward(inputs)
    np.testing.assert_almost_equal(output, expected_output)


def test_linear_layer_backward():
    input_size, output_size = 3, 2
    linear = Linear(input_size, output_size)

    # Set fixed weights for predictable testing
    linear.params["w"] = np.array([[1, 2], [3, 4], [5, 6]])
    linear.params["b"] = np.array([0.5, 1.5])

    inputs = np.array([[1, 2, 3]])
    linear.forward(inputs)

    grad = np.array([[2, 3]])

    # Test gradient computation
    backward_grad = linear.backward(grad)

    assert linear.grads["b"].shape == (2,)
    assert linear.grads["w"].shape == (3, 2)
    assert backward_grad.shape == (1, 3)


# Loss Tests
def test_mse_loss_calculation():
    mse = MSE()

    predicted = np.array([1, 2, 3])
    actual = np.array([0.5, 1.5, 2.5])

    expected_loss = np.mean((predicted - actual) ** 2)

    assert np.isclose(mse.loss(predicted, actual), expected_loss)


def test_mse_loss_gradient():
    mse = MSE()

    predicted = np.array([1, 2, 3])
    actual = np.array([0.5, 1.5, 2.5])

    expected_grad = 2 * (predicted - actual) / np.size(predicted)

    np.testing.assert_almost_equal(mse.grad(predicted, actual), expected_grad)


# Neural Network Tests
def test_sequential_net_forward():
    layers = [Linear(3, 4), Tanh(), Linear(4, 2)]
    net = SequentialNet(layers)

    inputs = np.array([[1, 2, 3]])
    output = net.forward(inputs)

    assert output.shape == (1, 2)


def test_sequential_net_backward():
    layers = [Linear(3, 4), Tanh(), Linear(4, 2)]
    net = SequentialNet(layers)

    inputs = np.array([[1, 2, 3]])
    net.forward(inputs)

    grad = np.array([[0.5, 0.5]])
    net.backward(grad)

    # Check that linear layers have gradients for their parameters
    for layer in net.layers:
        if isinstance(layer, Linear):
            assert "w" in layer.grads
            assert "b" in layer.grads
            assert layer.grads["w"].shape == layer.params["w"].shape
            assert layer.grads["b"].shape == layer.params["b"].shape


def test_params_and_grads():
    layers = [Linear(3, 4), Tanh(), Linear(4, 2)]
    net = SequentialNet(layers)

    inputs = np.array([[1, 2, 3]])
    net.forward(inputs)

    grad = np.array([[0.5, 0.5]])
    net.backward(grad)

    params_and_grads = list(net.params_and_grads())

    assert len(params_and_grads) > 0
    for param, grad in params_and_grads:
        assert isinstance(param, np.ndarray)
        assert isinstance(grad, np.ndarray)


# Optimizer Tests
def test_sgd_optimizer():
    layers = [Linear(3, 4), Tanh(), Linear(4, 2)]
    net = SequentialNet(layers)

    # Store initial weights
    initial_params = []
    for layer in net.layers:
        if hasattr(layer, "params"):
            layer_params = [param.copy() for param in layer.params.values()]
            initial_params.append(layer_params)

    inputs = np.array([[1, 2, 3]])
    net.forward(inputs)

    grad = np.array([[0.5, 0.5]])
    net.backward(grad)

    # Create SGD optimizer
    optimizer = SGD(lr=0.01)
    optimizer.step(net)

    # Check that weights have been updated
    for layer_idx, layer in enumerate(net.layers):
        if hasattr(layer, "params"):
            for param_idx, (name, param) in enumerate(layer.params.items()):
                initial_param = initial_params[layer_idx][param_idx]

                # Weights should have changed
                assert not np.array_equal(param, initial_param)


def test_sigmoid_activation():
    x = np.array([1, 10, 100])
    cor_f = np.array([0.731058578630074, 0.9999546021312978, 1])
    cor_f_prime = np.array([0.19661193324144993, 0.00004539580773568563, 0])

    o1 = Sigmoid().f(x)
    o2 = Sigmoid().f_prime(x)

    np.testing.assert_almost_equal(o1, cor_f)
    np.testing.assert_almost_equal(o2, cor_f_prime)
