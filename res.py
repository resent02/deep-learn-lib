import numpy as np

from deep_learn_lib.layers import Linear


def test_linear_layer_forward():
    # Test with known weights and biases
    linear = Linear(2, 3)
    linear.params["w"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    linear.params["b"] = np.array([0.1, 0.2, 0.3])

    inputs = np.array([[1.0, 2.0]])
    output = linear.forward(inputs)

    expected = np.array([[9.1, 12.2, 15.3]])  # 1*1 + 2*4 + 0.1, etc.

    assert np.allclose(output, expected), "Linear forward pass failed"


def test_linear_layer_backward():
    linear = Linear(2, 3)
    linear.params["w"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    linear.params["b"] = np.array([0.1, 0.2, 0.3])

    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])  # Batch of 2 samples

    # Forward pass to store inputs
    linear.forward(inputs)

    # Backward pass with dummy gradient
    grad_output = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    grad_input = linear.backward(grad_output)

    # Test weight gradients (inputs.T @ grad_output)
    expected_w_grad = np.array(
        [
            [
                1.3,
                1.6,
                1.9,
            ],  # (1.0*0.1 + 3.0*0.4, 1.0*0.2 + 3.0*0.5, 1.0*0.3 + 3.0*0.6)
            [
                2.6,
                3.2,
                3.8,
            ],  # (2.0*0.1 + 4.0*0.4, 2.0*0.2 + 4.0*0.5, 2.0*0.3 + 4.0*0.6)
        ]
    )
    assert np.allclose(linear.grads["w"], expected_w_grad), "Weight gradients incorrect"

    # Test bias gradients (sum(grad_output, axis=0))
    expected_b_grad = np.array([0.5, 0.7, 0.9])  # (0.1+0.4, 0.2+0.5, 0.3+0.6)
    assert np.allclose(linear.grads["b"], expected_b_grad), "Bias gradients incorrect"

    # Test input gradients (grad_output @ w.T)
    expected_input_grad = np.array(
        [
            [1.4, 3.2],  # (0.1*1.0 + 0.2*2.0 + 0.3*3.0, 0.1*4.0 + 0.2*5.0 + 0.3*6.0)
            [3.2, 7.4],  # (0.4*1.0 + 0.5*2.0 + 0.6*3.0, 0.4*4.0 + 0.5*5.0 + 0.6*6.0)
        ]
    )
    assert np.allclose(grad_input, expected_input_grad), "Input gradients incorrect"


test_linear_layer_backward()
