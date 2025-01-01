"""Tests for neural network components."""

import numpy as np

from tinytorch.engine import Tensor
from tinytorch.nn import MLP, Activation, Layer, Neuron


def test_neuron_forward():
    """Test neuron forward pass.

    Tests: different activations, forward and backward pass
    """
    # Create input
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
    x_tensor = Tensor(x)

    # Test different activations
    activations = [
        Activation.TANH,
        Activation.SIGMOID,
        Activation.RELU,
        Activation.LIN,
    ]

    for activation in activations:
        # Create neuron
        neuron = Neuron(n_input=3, activation=activation)

        # Forward pass
        output = neuron(x_tensor)

        # Backward pass
        output.sum().backward()

        # Check shapes
        assert output.shape == (2,)
        assert neuron.w.grad.shape == (1, 3)
        assert neuron.b.grad.shape == (1,)
        assert x_tensor.grad.shape == x_tensor.shape


def test_layer_forward():
    """Test layer forward pass.

    Tests: multiple neurons, different activations
    """
    # Create input
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)  # (3, 2)
    x_tensor = Tensor(x)

    # Test different activations
    activations = [
        Activation.TANH,
        Activation.SIGMOID,
        Activation.RELU,
        Activation.LIN,
    ]

    for activation in activations:
        # Create layer with 4 neurons
        layer = Layer(n_input=2, n_neurons=4, activation=activation)

        # Forward pass
        output = layer(x_tensor)

        # Backward pass
        output.sum().backward()

        # Check shapes
        assert output.shape == (3, 4)  # (batch_size, n_neurons)
        for neuron in layer.neurons:
            assert neuron.w.grad.shape == (1, 2)
            assert neuron.b.grad.shape == (1,)
        assert x_tensor.grad.shape == x_tensor.shape


def test_mlp_forward():
    """Test MLP forward pass.

    Tests: multiple layers, different activations
    """
    # Create input
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
    x_tensor = Tensor(x)

    # Define architecture: input_dim=3, hidden_dim=4, output_dim=2
    layers = [
        (4, Activation.TANH),  # Hidden layer
        (2, Activation.SIGMOID),  # Output layer
    ]

    # Create MLP
    mlp = MLP(n_input=3, layers=layers)

    # Forward pass
    output = mlp(x_tensor)

    # Backward pass
    output.sum().backward()

    # Check shapes
    assert output.shape == (2, 2)  # (batch_size, output_dim)

    # Check first layer shapes
    for neuron in mlp.layers[0].neurons:
        assert neuron.w.grad.shape == (1, 3)  # (input_dim, 1)
        assert neuron.b.grad.shape == (1,)

    # Check second layer shapes
    for neuron in mlp.layers[1].neurons:
        assert neuron.w.grad.shape == (1, 4)  # (hidden_dim, 1)
        assert neuron.b.grad.shape == (1,)

    assert x_tensor.grad.shape == x_tensor.shape
