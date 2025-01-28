"""Shared test configuration and utilities."""

import numpy as np
import pytest
import torch
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tinytorch.engine import Float32Array, Tensor


@pytest.fixture
def rtol() -> float:
    """Relative tolerance for float32."""
    return 1e-5


@pytest.fixture
def atol() -> float:
    """Absolute tolerance for float32."""
    return 1e-5


# Set test parameters
settings.register_profile("dev", max_examples=20)
settings.load_profile("dev")


@pytest.fixture
def default_floats_strategy() -> st.SearchStrategy:
    """Default strategy for floating point numbers."""
    return st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_infinity=False,
        allow_nan=False,
        width=32,
        allow_subnormal=False,
    )


@pytest.fixture
def default_ints_strategy() -> st.SearchStrategy:
    """Default strategy for integers."""
    return st.integers(min_value=-1000, max_value=1000)


@pytest.fixture
def default_lists_strategy(default_floats_strategy, default_ints_strategy) -> st.SearchStrategy:
    """Default strategy for lists."""
    return st.lists(default_floats_strategy | default_ints_strategy, min_size=1, max_size=10)


@pytest.fixture
def arrays_strategy(default_floats_strategy):
    """Strategy to generate numpy arrays."""

    def _arrays_strategy(
        max_dims: int = 4,
        max_size: int = 5,
        shape: tuple[int, ...] | None = None,
        floats_strategy=default_floats_strategy,
    ) -> st.SearchStrategy[Float32Array]:
        shape_strategy = (
            shape
            if shape is not None
            else st.lists(
                st.integers(min_value=1, max_value=max_size), min_size=1, max_size=max_dims
            )
        )
        return arrays(dtype=np.float32, shape=shape_strategy, elements=floats_strategy)  # type: ignore

    return _arrays_strategy


@pytest.fixture
def tensors_strategy(arrays_strategy):
    """Strategy to generate Tensor instances."""

    def _tensors_strategy(
        max_dims: int = 4,
        max_size: int = 5,
        shape: tuple[int, ...] | None = None,
        floats_strategy=None,  # Will use default from arrays_strategy
    ) -> st.SearchStrategy[Tensor]:
        return arrays_strategy(max_dims, max_size, shape, floats_strategy).map(Tensor)

    return _tensors_strategy


@pytest.fixture
def same_shape_tensors_strategy(arrays_strategy):
    """Strategy to generate two tensors with the same shape."""

    def _same_shape_tensors_strategy(
        max_dims: int = 4,
        max_size: int = 5,
        floats_strategy=None,  # Will use default from arrays_strategy
    ) -> st.SearchStrategy[tuple[Tensor, Tensor]]:
        @st.composite
        def two_tensors(draw):
            shape = draw(
                st.lists(
                    st.integers(min_value=1, max_value=max_size),
                    min_size=1,
                    max_size=max_dims,
                )
            )
            t1 = draw(arrays_strategy(shape=shape, floats_strategy=floats_strategy))
            t2 = draw(arrays_strategy(shape=shape, floats_strategy=floats_strategy))
            return Tensor(t1), Tensor(t2)

        return two_tensors()

    return _same_shape_tensors_strategy


@pytest.fixture
def torch_tensor_strategy(arrays_strategy):
    """Strategy to generate PyTorch tensors."""

    def _torch_tensor_strategy(
        max_dims: int = 4,
        max_size: int = 5,
        shape: tuple[int, ...] | None = None,
        floats_strategy=None,  # Will use default from arrays_strategy
    ) -> st.SearchStrategy[torch.Tensor]:
        array_strat = arrays_strategy(max_dims, max_size, shape, floats_strategy)
        return array_strat.map(torch.Tensor)

    return _torch_tensor_strategy
