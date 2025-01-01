"""Tests for backpropagation using PyTorch for verification."""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import ATOL, RTOL, same_shape_tensors_strategy, tensors_strategy
from tinytorch.engine import Tensor


@given(same_shape_tensors_strategy())
@settings(deadline=None)
def test_add_gradients(tensors):
    """Test addition gradients.

    Tests: gradient computation against PyTorch
    """
    x, y = tensors
    z = x + y
    z.backward()

    # Create PyTorch tensors from our tensor's data
    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = torch.tensor(y.data, requires_grad=True)
    z_torch = x_torch + y_torch
    z_torch.backward(torch.ones_like(z_torch))

    # Compare gradients
    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(y.grad, y_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(same_shape_tensors_strategy())
def test_mul_gradients(tensors):
    """Test multiplication gradients.

    Tests: gradient computation against PyTorch
    """
    x, y = tensors
    z = x * y
    z.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = torch.tensor(y.data, requires_grad=True)
    z_torch = x_torch * y_torch
    z_torch.backward(torch.ones_like(z_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(y.grad, y_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(same_shape_tensors_strategy())
def test_sub_gradients(tensors):
    """Test subtraction gradients.

    Tests: gradient computation against PyTorch
    """
    x, y = tensors
    z = x - y
    z.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = torch.tensor(y.data, requires_grad=True)
    z_torch = x_torch - y_torch
    z_torch.backward(torch.ones_like(z_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(y.grad, y_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_exp_gradients(tensor):
    """Test exponential gradients.

    Tests: gradient computation against PyTorch
    """
    x = tensor
    y = x.exp()
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = torch.exp(x_torch)
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(same_shape_tensors_strategy())
def test_max_gradients(tensors):
    """Test maximum gradients.

    Tests: gradient computation against PyTorch
    """
    x, y = tensors
    z = x.max(y)
    z.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = torch.tensor(y.data, requires_grad=True)
    z_torch = torch.maximum(x_torch, y_torch)
    z_torch.backward(torch.ones_like(z_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(y.grad, y_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_relu_gradients(tensor):
    """Test ReLU gradients.

    Tests: gradient computation against PyTorch
    """
    x = tensor
    y = x.relu()
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = torch.relu(x_torch)
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_chained_ops_gradients(tensor):
    """Test chained operation gradients.

    Tests: x^2 + exp(x) against PyTorch
    """
    x = tensor
    y = x * x + x.exp()
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = x_torch * x_torch + torch.exp(x_torch)
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_self_mul_gradients(tensor):
    """Test self-multiplication gradients.

    Tests: x * x gradient against PyTorch
    """
    x = tensor
    y = x * x  # Should give gradient of 2x
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = x_torch * x_torch
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_self_add_gradients(tensor):
    """Test self-addition gradients.

    Tests: x + x gradient against PyTorch
    """
    x = tensor
    y = x + x  # Should give gradient of 2
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = x_torch + x_torch
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_self_chain_gradients(tensor):
    """Test self-chained gradients.

    Tests: x^2 + x gradient against PyTorch
    """
    x = tensor
    y = (x * x) + x  # Should give gradient of 2x + 1
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = (x_torch * x_torch) + x_torch
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_broadcast_add_gradients(tensor):
    """Test broadcast addition gradients.

    Tests: tensor + scalar against PyTorch
    """
    x = tensor
    scalar = Tensor(2.0)  # scalar will be broadcast
    y = x + scalar
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    scalar_torch = torch.tensor(2.0, requires_grad=True)
    y_torch = x_torch + scalar_torch
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(scalar.grad, scalar_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_broadcast_mul_gradients(tensor):
    """Test broadcast multiplication gradients.

    Tests: tensor * scalar against PyTorch
    """
    x = tensor
    scalar = Tensor(3.0)  # scalar will be broadcast
    y = x * scalar
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    scalar_torch = torch.tensor(3.0, requires_grad=True)
    y_torch = x_torch * scalar_torch
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(scalar.grad, scalar_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


def test_broadcast_specific_shapes():
    """Test broadcast shape handling.

    Tests: scalar-vector, vector-matrix, scalar-matrix
    """
    # Test broadcasting scalar to vector
    x = Tensor([1.0, 2.0, 3.0])
    scalar = Tensor(2.0)
    y = x * scalar
    y.backward()
    assert scalar.grad.shape == ()
    assert scalar.grad.item() == 6.0  # sum of x's elements
    assert np.array_equal(x.grad, np.array([2.0, 2.0, 2.0]))

    # Test broadcasting vector to matrix
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    v = Tensor([1.0, 2.0])  # (2,) -> (1, 2)
    y = x + v
    y.backward()
    assert v.grad.shape == (2,)
    assert np.array_equal(v.grad, np.array([2.0, 2.0]))  # summed over broadcasted dimension
    assert np.array_equal(x.grad, np.ones_like(x.data))

    # Test broadcasting scalar to matrix
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    scalar = Tensor(2.0)  # (1,) -> (2, 2)
    y = x * scalar
    y.backward()
    assert scalar.grad.shape == ()
    assert scalar.grad.item() == 10.0  # sum of x's elements
    assert np.array_equal(x.grad, np.full_like(x.data, 2.0))


def test_broadcast_chain_operations():
    """Test broadcast chain gradients.

    Tests: (x * v) + s with different shapes
    """
    # Create tensors of different shapes
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    v = Tensor([0.5, 1.0])  # (2,)
    s = Tensor(2.0)  # scalar

    # Chain of operations with broadcasting
    y = (x * v) + s  # v is broadcast to (2, 2), then s is broadcast to (2, 2)
    y.backward()

    # Verify gradients with PyTorch
    x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    v_torch = torch.tensor([0.5, 1.0], requires_grad=True)
    s_torch = torch.tensor(2.0, requires_grad=True)
    y_torch = (x_torch * v_torch) + s_torch
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(v.grad, v_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s.grad, s_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_multiple_use_gradients(tensor):
    """Test multiple use gradients.

    Tests: x^2 + x gradient accumulation
    """
    x = tensor
    y = x * x + x  # Uses x twice: once in x*x and once in +x
    y.backward()

    # Verify against PyTorch
    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = x_torch * x_torch + x_torch
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_neg_gradients(tensor):
    """Test negation gradients.

    Tests: -x gradient against PyTorch
    """
    x = tensor
    y = -x  # Should give gradient of -1
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = -x_torch
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy(), st.integers(2, 5))
def test_pow_gradients(tensor, exponent):
    """Test power gradients.

    Tests: x^n gradient against PyTorch
    """
    x = tensor
    y = x**exponent  # Should give gradient of n * x^(n-1)
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = x_torch**exponent
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(same_shape_tensors_strategy())
def test_min_gradients(tensors):
    """Test minimum gradients.

    Tests: gradient computation against PyTorch
    """
    x, y = tensors
    z = x.min(y)
    z.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = torch.tensor(y.data, requires_grad=True)
    z_torch = torch.minimum(x_torch, y_torch)
    z_torch.backward(torch.ones_like(z_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(y.grad, y_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_sum_gradients(tensor):
    """Test sum gradients.

    Tests: gradient computation against PyTorch
    """
    x = tensor
    y = x.sum()  # Sum over all axes
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = x_torch.sum()
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


def test_stack_gradients():
    """Test stack gradients.

    Tests: simple stack+sum, complex stack+multiply
    """
    # Test 1: Simple stacking and sum
    t1 = Tensor([1.0, 2.0])
    t2 = Tensor([3.0, 4.0])
    t3 = Tensor([5.0, 6.0])

    # Create corresponding PyTorch tensors
    t1_torch = torch.tensor([1.0, 2.0], requires_grad=True)
    t2_torch = torch.tensor([3.0, 4.0], requires_grad=True)
    t3_torch = torch.tensor([5.0, 6.0], requires_grad=True)

    # Stack and sum in both frameworks
    stacked = Tensor.stack([t1, t2, t3], axis=0)  # shape: (3, 2)
    result = stacked.sum()

    stacked_torch = torch.stack([t1_torch, t2_torch, t3_torch], dim=0)
    result_torch = stacked_torch.sum()

    # Backpropagate
    result.backward()
    result_torch.backward()

    # Compare gradients
    np.testing.assert_allclose(t1.grad, t1_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(t2.grad, t2_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(t3.grad, t3_torch.grad.numpy(), rtol=RTOL, atol=ATOL)

    # Test 2: Stack along different axis and more complex computation
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    t2 = Tensor([[5.0, 6.0], [7.0, 8.0]])

    t1_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    t2_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # Stack along last axis, multiply by 2, and sum
    stacked = Tensor.stack([t1, t2], axis=2)  # shape: (2, 2, 2)
    result = (stacked * 2.0).sum()

    stacked_torch = torch.stack([t1_torch, t2_torch], dim=2)
    result_torch = (stacked_torch * 2.0).sum()

    # Backpropagate
    result.backward()
    result_torch.backward()

    # Compare gradients
    np.testing.assert_allclose(t1.grad, t1_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(t2.grad, t2_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_lin_gradients(tensor):
    """Test linear activation gradients.

    Tests: gradient computation against PyTorch
    """
    x = tensor
    y = x.lin()  # Should give gradient of 1
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = x_torch  # Identity function
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)


@given(tensors_strategy())
def test_log_gradients(tensor):
    """Test logarithm gradients.

    Tests: gradient computation against PyTorch
    """
    # Ensure input is positive to avoid log(negative)
    x = Tensor(np.abs(tensor.data) + 1e-6)  # Add small epsilon to avoid log(0)
    y = x.log()
    y.backward()

    x_torch = torch.tensor(x.data, requires_grad=True)
    y_torch = torch.log(x_torch)
    y_torch.backward(torch.ones_like(y_torch))

    np.testing.assert_allclose(x.grad, x_torch.grad.numpy(), rtol=RTOL, atol=ATOL)
