"""Tests for arithmetic operations."""

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st

from tinytorch.engine import Operation, Tensor


def test_add_operation(same_shape_tensors_strategy):
    """Test tensor addition.

    Tests: operation type, children tracking, scalar addition, numpy equivalence
    """

    @given(same_shape_tensors_strategy())
    def _test(tensors: tuple[Tensor, Tensor]):
        t1, t2 = tensors
        note(f"Testing shapes: {t1.shape} and {t2.shape}")

        # Test tensor-tensor addition
        forward = t1 + t2
        assert forward._op == Operation.ADD, "Addition operation type should be ADD"
        assert forward._children == {
            t1,
            t2,
        }, "Addition should track both operands as children"
        np.testing.assert_array_equal(
            forward.data, t1.data + t2.data, "Tensor addition should match numpy addition"
        )

        # Test scalar-tensor addition (tests __radd__)
        reverse = 1.0 + t1
        assert reverse._op == Operation.ADD, "Scalar addition should use ADD operation"
        np.testing.assert_array_equal(
            reverse.data,
            1.0 + t1.data,
            "Scalar addition should match numpy scalar addition",
        )

    _test()


def test_multiply_operation(same_shape_tensors_strategy):
    """Test multiplication operations between tensors and with scalars.

    Properties tested:
    1. Basic tensor multiplication: t1 * t2
    2. Operation type is correctly set
    3. Children are tracked properly
    4. Scalar multiplication via __rmul__
    5. Numerical correctness against numpy
    """

    @given(same_shape_tensors_strategy())
    def _test(tensors: tuple[Tensor, Tensor]):
        t1, t2 = tensors
        note(f"Testing shapes: {t1.shape} and {t2.shape}")

        # Test tensor-tensor multiplication
        forward = t1 * t2
        assert forward._op == Operation.MULT, "Multiplication operation type should be MULT"
        assert forward._children == {
            t1,
            t2,
        }, "Multiplication should track both operands as children"
        np.testing.assert_array_equal(
            forward.data,
            t1.data * t2.data,
            "Tensor multiplication should match numpy multiplication",
        )

        # Test scalar-tensor multiplication (tests __rmul__)
        reverse = 2.0 * t1
        assert reverse._op == Operation.MULT, "Scalar multiplication should use MULT operation"
        np.testing.assert_array_equal(
            reverse.data,
            2.0 * t1.data,
            "Scalar multiplication should match numpy scalar multiplication",
        )

    _test()


def test_subtract_operation(same_shape_tensors_strategy):
    """Test subtraction operations between tensors and with scalars.

    Properties tested:
    1. Basic tensor subtraction: t1 - t2
    2. Operation type is correctly set
    3. Scalar subtraction via __rsub__
    4. Numerical correctness against numpy
    """

    @given(same_shape_tensors_strategy())
    def _test(tensors: tuple[Tensor, Tensor]):
        t1, t2 = tensors
        note(f"Testing shapes: {t1.shape} and {t2.shape}")

        # Test tensor-tensor subtraction
        forward = t1 - t2
        np.testing.assert_array_equal(
            forward.data,
            t1.data - t2.data,
            "Tensor subtraction should match numpy subtraction",
        )

        # Test scalar-tensor subtraction (tests __rsub__)
        reverse = 1.0 - t1
        np.testing.assert_array_equal(
            reverse.data,
            1.0 - t1.data,
            "Scalar subtraction should match numpy scalar subtraction",
        )

    _test()


def test_add_commutative(same_shape_tensors_strategy):
    """Test that a + b = b + a"""

    @given(same_shape_tensors_strategy())
    def _test(tensors: tuple[Tensor, Tensor]):
        t1, t2 = tensors
        note(f"Testing shapes: {t1.shape} and {t2.shape}")
        forward = t1 + t2
        reverse = t2 + t1
        np.testing.assert_array_equal(forward.data, reverse.data)
        assert forward._children == reverse._children

    _test()


def test_multiply_commutative(same_shape_tensors_strategy):
    """Test multiplication commutativity.

    Tests: a * b == b * a, children tracking
    """

    @given(same_shape_tensors_strategy())
    def _test(tensors: tuple[Tensor, Tensor]):
        t1, t2 = tensors
        note(f"Testing shapes: {t1.shape} and {t2.shape}")
        forward = t1 * t2
        reverse = t2 * t1
        np.testing.assert_array_equal(forward.data, reverse.data)
        assert forward._children == reverse._children

    _test()


def test_divide_operation(same_shape_tensors_strategy, rtol, atol):
    """Test tensor division.

    Tests: operation type, (a/b) * b ≈ a property, numpy equivalence
    """

    @given(
        same_shape_tensors_strategy(
            floats_strategy=st.floats(
                min_value=0.0010000000474974513,
                max_value=10,
                allow_infinity=False,
                allow_nan=False,
                exclude_min=True,
                width=32,
                allow_subnormal=False,
            )
        )
    )
    def _test(tensors: tuple[Tensor, Tensor]):
        t1, t2 = tensors
        note(f"Testing shapes: {t1.shape} and {t2.shape}")

        # Test tensor-tensor division
        forward = t1 / t2

        # Test division-multiplication relationship
        result = forward * t2
        np.testing.assert_allclose(
            result.data,
            t1.data,
            rtol=rtol,
            atol=atol,
            err_msg="Division-multiplication property: (a/b) * b should approximately equal a",
        )

    _test()


def test_power_operation(tensors_strategy):
    """Test tensor exponentiation.

    Tests: operation type, children tracking, numpy equivalence
    """

    @given(tensors_strategy(), st.integers(2, 5))
    def _test(t1: Tensor, exponent: int):
        result = t1**exponent
        assert result._op == Operation.POW
        assert result._children == {t1}
        np.testing.assert_array_equal(result.data, np.power(t1.data, exponent))

    _test()


def test_add_with_scalar(tensors_strategy, default_floats_strategy):
    """Test scalar addition.

    Tests: operation type, commutativity, numpy equivalence
    """

    @given(tensors_strategy(), default_floats_strategy)
    def _test(tensor: Tensor, scalar: float):
        result1 = tensor + scalar
        result2 = scalar + tensor
        assert result1._op == Operation.ADD
        assert result2._op == Operation.ADD
        np.testing.assert_array_equal(result1.data, tensor.data + scalar)
        np.testing.assert_array_equal(result2.data, scalar + tensor.data)
        np.testing.assert_array_equal(result1.data, result2.data)  # Commutativity

    _test()


def test_multiply_with_scalar(tensors_strategy, default_floats_strategy):
    """Test scalar multiplication.

    Tests: operation type, commutativity, numpy equivalence
    """

    @given(tensors_strategy(), default_floats_strategy)
    def _test(tensor: Tensor, scalar: float):
        result1 = tensor * scalar
        result2 = scalar * tensor
        assert result1._op == Operation.MULT
        assert result2._op == Operation.MULT
        np.testing.assert_array_equal(result1.data, tensor.data * scalar)
        np.testing.assert_array_equal(result2.data, scalar * tensor.data)
        np.testing.assert_array_equal(result1.data, result2.data)  # Commutativity

    _test()


def test_subtract_with_scalar(tensors_strategy, default_floats_strategy):
    """Test subtraction between tensor and scalar."""

    @given(tensors_strategy(), default_floats_strategy)
    def _test(tensor: Tensor, scalar: float):
        result1 = tensor - scalar
        result2 = scalar - tensor
        np.testing.assert_array_equal(result1.data, tensor.data - scalar)
        np.testing.assert_array_equal(result2.data, scalar - tensor.data)

    _test()


def test_divide_with_scalar(tensors_strategy, rtol, atol):
    """Test division between tensor and scalar."""

    @given(
        tensors_strategy(),
        st.floats(
            min_value=0.0010000000474974513,
            max_value=10,
            allow_infinity=False,
            allow_nan=False,
            exclude_min=True,
            width=32,
            allow_subnormal=False,
        ),
    )
    def _test(tensor: Tensor, scalar: float):
        result1 = tensor / scalar
        result2 = scalar / tensor
        np.testing.assert_allclose(result1.data, tensor.data / scalar, rtol=rtol, atol=atol)
        np.testing.assert_allclose(result2.data, scalar / tensor.data, rtol=rtol, atol=atol)

    _test()


def test_matmul_operation():
    """Test matrix multiplication.

    Tests: matrix-matrix, matrix-vector, vector-vector ops, shape validation
    """
    # Test matrix-matrix multiplication
    m1 = Tensor([[1.0, 2.0], [3.0, 4.0]])  # 2x2
    m2 = Tensor([[5.0, 6.0], [7.0, 8.0]])  # 2x2
    result = m1 @ m2
    assert result._op == Operation.MATMUL
    assert result._children == {m1, m2}
    np.testing.assert_array_equal(result.data, np.matmul(m1.data, m2.data))

    # Test matrix-vector multiplication
    v = Tensor([1.0, 2.0])  # 2x1
    result = m1 @ v
    assert result._op == Operation.MATMUL
    np.testing.assert_array_equal(result.data, np.matmul(m1.data, v.data))

    # Test vector-vector multiplication (dot product)
    v1 = Tensor([1.0, 2.0])
    v2 = Tensor([3.0, 4.0])
    result = v1 @ v2
    assert result._op == Operation.MATMUL
    assert result.shape == (), "Dot product should return a scalar"
    np.testing.assert_array_equal(result.data, np.matmul(v1.data, v2.data))
    assert result.data.item() == 11.0  # 1*3 + 2*4 = 11


def test_sum_operation(tensors_strategy):
    """Test tensor summation.

    Tests: all axes, specific axis, multiple axes, numpy equivalence
    """

    @given(tensors_strategy())
    def _test(tensor: Tensor):
        note(f"Testing shape: {tensor.shape}")

        # Test sum over all axes
        result = tensor.sum()
        assert result._op == Operation.SUM
        assert result._children == {tensor}
        np.testing.assert_array_equal(
            result.data,
            np.sum(tensor.data),
            "Tensor sum should match numpy sum",
        )

        # Test sum over first axis if tensor has multiple dimensions
        if len(tensor.shape) > 1:
            result = tensor.sum(axis=0)
            assert result._op == Operation.SUM
            assert result._children == {tensor}
            np.testing.assert_array_equal(
                result.data,
                np.sum(tensor.data, axis=0),
                "Tensor sum over axis should match numpy sum",
            )

        # Test sum over multiple axes if tensor has at least 3 dimensions
        if len(tensor.shape) >= 3:
            axes = (0, 2)
            result = tensor.sum(axis=axes)
            assert result._op == Operation.SUM
            assert result._children == {tensor}
            np.testing.assert_array_equal(
                result.data,
                np.sum(tensor.data, axis=axes),
                "Tensor sum over multiple axes should match numpy sum",
            )

    _test()


def test_stack_operation(atol, rtol):
    """Test tensor stacking.

    Tests: 1D/2D tensors, different axes, shape validation
    """
    # Test 1D tensors
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    t3 = Tensor([7, 8, 9])

    # Stack along axis 0 (default)
    result = Tensor.stack([t1, t2, t3])
    expected = np.stack([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_allclose(result.data, expected, rtol=rtol, atol=atol)

    # Stack along axis 1
    result = Tensor.stack([t1, t2, t3], axis=1)
    expected = np.stack([[1, 2, 3], [4, 5, 6], [7, 8, 9]], axis=1)
    np.testing.assert_allclose(result.data, expected, rtol=rtol, atol=atol)

    # Test 2D tensors
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])

    # Stack along axis 0
    result = Tensor.stack([t1, t2])
    expected = np.stack([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    np.testing.assert_allclose(result.data, expected, rtol=rtol, atol=atol)

    # Stack along axis 2
    result = Tensor.stack([t1, t2], axis=2)
    expected = np.stack([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], axis=2)
    np.testing.assert_allclose(result.data, expected, rtol=rtol, atol=atol)


def test_log(tensors_strategy, rtol, atol):
    """Test natural logarithm operation.

    Tests: forward pass computation against numpy
    """

    @given(tensors_strategy())
    def _test(tensor: Tensor):
        # Ensure input is positive to avoid log(negative)
        x = Tensor(
            (np.abs(tensor.data) + 1e-6).astype(np.float32)
        )  # Add small epsilon to avoid log(0)
        result = x.log()
        expected = np.log(x.data)
        np.testing.assert_allclose(result.data, expected, rtol=rtol, atol=atol)

    _test()
