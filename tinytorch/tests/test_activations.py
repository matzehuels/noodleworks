"""Tests for neural network activation functions."""

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from tinytorch.engine import Operation, Tensor


def test_max_operation(same_shape_tensors_strategy):
    """Test element-wise maximum.

    Tests: operation type, numpy equivalence
    """

    @given(same_shape_tensors_strategy())
    def _test(tensors: tuple[Tensor, Tensor]):
        t1, t2 = tensors
        result = t1.max(t2)
        assert result._op == Operation.MAX
        np.testing.assert_array_equal(result.data, np.maximum(t1.data, t2.data))

    _test()


def test_lin_operation(tensors_strategy):
    """Test linear activation.

    Tests: operation type, children tracking, identity property
    """

    @given(tensors_strategy())
    def _test(tensor):
        result = tensor.lin()
        assert result._op == Operation.IDENT
        assert result._children == {tensor}
        np.testing.assert_array_equal(
            result.data,
            tensor.data,
            "Linear activation should match input exactly",
        )

    _test()


def test_max_with_scalar(tensors_strategy, default_floats_strategy):
    """Test scalar maximum.

    Tests: operation type, numpy equivalence
    """

    @given(tensors_strategy(), default_floats_strategy)
    def _test(tensor: Tensor, scalar: float):
        result = tensor.max(scalar)
        assert result._op == Operation.MAX
        np.testing.assert_array_equal(result.data, np.maximum(tensor.data, scalar))

    _test()


def test_relu_operation(tensors_strategy):
    """Test ReLU activation.

    Tests: max(0,x) property, numpy equivalence
    """

    @given(tensors_strategy())
    def _test(tensor: Tensor):
        result = tensor.relu()
        np.testing.assert_array_equal(result.data, np.maximum(tensor.data, 0))

    _test()


def test_exp_operation(tensors_strategy):
    """Test exponential function.

    Tests: operation type, children tracking, numpy equivalence
    """

    @given(tensors_strategy())
    def _test(t1: Tensor):
        result = t1.exp()
        assert result._op == Operation.EXP
        assert result._children == {t1}
        np.testing.assert_array_equal(result.data, np.exp(t1.data))

    _test()


def test_sigmoid_operation(tensors_strategy, rtol, atol):
    """Test sigmoid activation.

    Tests: 1/(1+e^(-x)) formula, numpy equivalence
    """

    @given(tensors_strategy())
    def _test(tensor: Tensor):
        result = tensor.sigmoid()
        np.testing.assert_allclose(
            result.data, 1 / (1 + np.exp(-tensor.data)), rtol=rtol, atol=atol
        )

    _test()


def test_tanh_operation(tensors_strategy, rtol, atol):
    """Test hyperbolic tangent.

    Tests: tanh formula, numpy equivalence
    """

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
    def _test(tensor: Tensor):
        result = tensor.tanh()
        np.testing.assert_allclose(result.data, np.tanh(tensor.data), rtol=rtol, atol=atol)

    _test()
