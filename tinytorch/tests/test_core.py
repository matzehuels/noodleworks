"""Tests for core Tensor functionality."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tinytorch.engine import Operation, Tensor


def test_constructor_and_core_properties(arrays_strategy):
    """Test tensor construction.

    Tests: data type, shape, repr
    """

    @given(arrays_strategy())
    def _test(array: np.ndarray):
        tensor = Tensor(data=array)
        assert isinstance(tensor, Tensor)
        assert isinstance(tensor.data, np.ndarray)
        assert isinstance(tensor.shape, tuple)
        assert tensor.shape == array.shape
        assert repr(tensor) == f"Tensor(shape={array.shape})"

    _test()


def test_negation_operation(tensors_strategy):
    """Test tensor negation.

    Tests: double negation, -1 multiplication, zero handling
    """

    @given(tensors_strategy())
    def _test(tensor: Tensor):
        # Test double negation
        double_neg = -1 * (-tensor)
        np.testing.assert_array_equal(double_neg.data, tensor.data)

        # Test negation is same as multiplying by -1
        neg = -tensor
        mult_neg = tensor * -1
        np.testing.assert_array_equal(neg.data, mult_neg.data)
        assert neg._op == Operation.MULT  # Negation uses multiplication internally

        # Test negating zero returns zero
        zero_tensor = Tensor(np.zeros_like(tensor.data, dtype=np.float32))
        np.testing.assert_array_equal((-zero_tensor).data, zero_tensor.data)

    _test()


def test_tensor_labeling(arrays_strategy):
    """Test tensor labeling.

    Tests: label storage, None handling
    """

    @given(arrays_strategy(), st.text(min_size=1))
    def _test(array: np.ndarray, label: str):
        tensor = Tensor(data=array)
        assert tensor.label is None

        # Test label can be set
        labeled = Tensor(data=array, label=label)
        assert labeled.label == label

    _test()


def test_repr_equals_str(tensors_strategy):
    """Test string representations.

    Tests: __repr__ == __str__, labeled tensors
    """

    @given(tensors_strategy())
    def _test(tensor):
        assert str(tensor) == repr(tensor)

        # Also test with a labeled tensor
        labeled = Tensor(tensor.data.astype(np.float32), label="test")
        assert str(labeled) == repr(labeled)

    _test()


def test_len():
    """Test __len__ behavior matches numpy.

    Tests:
    1. Scalar tensors raise TypeError
    2. Vector length is first dimension
    3. Matrix length is first dimension
    4. Multi-dimensional tensor length is first dimension
    """
    # Test scalar (0-d tensor)
    scalar_tensor = Tensor(5.0)
    with pytest.raises(TypeError):
        len(scalar_tensor)

    # Test vector (1-d tensor)
    vector_tensor = Tensor([1.0, 2.0, 3.0])
    vector_numpy = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert len(vector_tensor) == len(vector_numpy) == 3

    # Test matrix (2-d tensor)
    matrix_tensor = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    matrix_numpy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    assert len(matrix_tensor) == len(matrix_numpy) == 3

    # Test higher dimensions (3-d tensor)
    tensor_3d = Tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    numpy_3d = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]], dtype=np.float32)
    assert len(tensor_3d) == len(numpy_3d) == 4
