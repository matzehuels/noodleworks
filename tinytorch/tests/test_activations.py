"""Tests for neural network activation functions."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from tests.conftest import (
    ATOL,
    RTOL,
    default_floats_strategy,
    same_shape_tensors_strategy,
    tensors_strategy,
)
from tinytorch.engine import Operation, Tensor


@given(same_shape_tensors_strategy())
@settings(deadline=None)
def test_max_operation(tensors: tuple[Tensor, Tensor]) -> None:
    """Test element-wise maximum.

    Tests: operation type, numpy equivalence
    """
    t1, t2 = tensors
    result = t1.max(t2)
    assert result._op == Operation.MAX
    np.testing.assert_array_equal(result.data, np.maximum(t1.data, t2.data))


@given(tensors_strategy())
def test_lin_operation(tensor):
    """Test linear activation.

    Tests: operation type, children tracking, identity property
    """
    result = tensor.lin()
    assert result._op == Operation.IDENT
    assert result._children == {tensor}
    np.testing.assert_array_equal(
        result.data,
        tensor.data,
        "Linear activation should match input exactly",
    )


@given(tensors_strategy(), default_floats_strategy)
def test_max_with_scalar(tensor: Tensor, scalar: float) -> None:
    """Test scalar maximum.

    Tests: operation type, numpy equivalence
    """
    result = tensor.max(scalar)
    assert result._op == Operation.MAX
    np.testing.assert_array_equal(result.data, np.maximum(tensor.data, scalar))


@given(tensors_strategy())
def test_relu_operation(tensor: Tensor) -> None:
    """Test ReLU activation.

    Tests: max(0,x) property, numpy equivalence
    """
    result = tensor.relu()
    np.testing.assert_array_equal(result.data, np.maximum(tensor.data, 0))


@given(tensors_strategy())
def test_exp_operation(t1: Tensor) -> None:
    """Test exponential function.

    Tests: operation type, children tracking, numpy equivalence
    """
    result = t1.exp()
    assert result._op == Operation.EXP
    assert result._children == {t1}
    np.testing.assert_array_equal(result.data, np.exp(t1.data))


@given(tensors_strategy())
def test_sigmoid_operation(tensor: Tensor) -> None:
    """Test sigmoid activation.

    Tests: 1/(1+e^(-x)) formula, numpy equivalence
    """
    result = tensor.sigmoid()
    np.testing.assert_allclose(result.data, 1 / (1 + np.exp(-tensor.data)), rtol=RTOL, atol=ATOL)


@given(
    tensors_strategy(
        floats_strategy=st.floats(
            min_value=0.0010000000474974513,
            max_value=10,
            allow_infinity=False,
            allow_nan=False,
            allow_subnormal=False,
            width=32,
        )
    )
)
def test_tanh_operation(tensor: Tensor) -> None:
    """Test hyperbolic tangent.

    Tests: tanh formula, numpy equivalence
    """
    result = tensor.tanh()
    np.testing.assert_allclose(result.data, np.tanh(tensor.data), rtol=RTOL, atol=ATOL)
