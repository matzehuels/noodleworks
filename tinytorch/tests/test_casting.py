"""Tests for type casting functionality."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tinytorch.engine import ArrayLike, Tensor, TensorLike, _cast_array, _cast_tensor


def test_cast_array_invalid_input():
    """Test array casting errors.

    Tests: TypeError for invalid inputs
    """

    @given(
        st.one_of(
            st.dictionaries(st.text(), st.integers()),
            st.text(),
            st.binary(),
            st.none(),
        )
    )
    def _test(invalid_data):
        with pytest.raises(TypeError):
            _cast_array(invalid_data)

    _test()


def test_cast_tensor_invalid_input():
    """Test tensor casting errors.

    Tests: TypeError for invalid inputs
    """

    @given(
        st.one_of(
            st.dictionaries(st.text(), st.integers()),
            st.text(),
            st.binary(),
            st.none(),
        )
    )
    def _test(invalid_data):
        with pytest.raises(TypeError):
            _cast_tensor(invalid_data)

    _test()


def test_cast_array(
    arrays_strategy, default_floats_strategy, default_ints_strategy, default_lists_strategy
):
    """Test array casting.

    Tests: type conversion, shape preservation
    """

    @given(
        st.one_of(
            arrays_strategy(),
            default_floats_strategy,
            default_ints_strategy,
            default_lists_strategy,
        )
    )
    def _test(data: ArrayLike):
        result = _cast_array(data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

        if isinstance(data, (float, int)):
            assert result.shape == ()
        elif isinstance(data, list):
            assert result.shape == (len(data),)
        elif isinstance(data, np.ndarray):
            assert result.shape == data.shape

    _test()


def test_cast_tensor(
    arrays_strategy,
    tensors_strategy,
    default_floats_strategy,
    default_ints_strategy,
    default_lists_strategy,
):
    """Test tensor casting.

    Tests: type conversion, shape preservation
    """

    @given(
        st.one_of(
            arrays_strategy(),
            tensors_strategy(),
            default_floats_strategy,
            default_ints_strategy,
            default_lists_strategy,
        )
    )
    def _test(data: TensorLike):
        result = _cast_tensor(data)
        assert isinstance(result, Tensor)
        assert result.data.dtype == np.float32

        if isinstance(data, (float, int)):
            assert result.shape == ()
        elif isinstance(data, list):
            assert result.shape == (len(data),)
        elif isinstance(data, np.ndarray):
            assert result.shape == data.shape
        elif isinstance(data, Tensor):
            assert result.shape == data.shape

    _test()
