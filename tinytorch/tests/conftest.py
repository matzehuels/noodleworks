"""Shared test configuration and utilities."""

from typing import Optional, Tuple

import numpy as np
import torch  # type: ignore
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tinytorch.engine import Float32Array, Tensor

# Tolerances
RTOL = 1e-5  # relative tolerance for float32
ATOL = 1e-5  # absolute tolerance for float32

# Set test parameters
settings.register_profile("dev", max_examples=20)
settings.load_profile("dev")

# Default strategies
default_floats_strategy = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_infinity=False,
    allow_nan=False,
    width=32,
    allow_subnormal=False,
)
default_ints_strategy = st.integers(min_value=-1000, max_value=1000)
default_lists_strategy = st.lists(
    default_floats_strategy | default_ints_strategy, min_size=1, max_size=10
)


def arrays_strategy(
    max_dims: int = 4,
    max_size: int = 5,
    shape: Optional[Tuple[int, ...]] = None,
    floats_strategy=default_floats_strategy,
) -> st.SearchStrategy[Float32Array]:
    """Strategy to generate numpy arrays."""
    shape_strategy = (
        shape
        if shape is not None
        else st.lists(st.integers(min_value=1, max_value=max_size), min_size=1, max_size=max_dims)
    )
    return arrays(dtype=np.float32, shape=shape_strategy, elements=floats_strategy)  # type: ignore


def tensors_strategy(
    max_dims: int = 4,
    max_size: int = 5,
    shape: Optional[Tuple[int, ...]] = None,
    floats_strategy=default_floats_strategy,
) -> st.SearchStrategy[Tensor]:
    """Generates Tensor instances with random numpy arrays."""
    return arrays_strategy(max_dims, max_size, shape, floats_strategy).map(Tensor)


def same_shape_tensors_strategy(
    max_dims: int = 4, max_size: int = 5, floats_strategy=default_floats_strategy
) -> st.SearchStrategy[Tuple[Tensor, Tensor]]:
    """Strategy to generate two tensors with the same shape."""

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


def torch_tensor_strategy(
    max_dims: int = 4,
    max_size: int = 5,
    shape: Optional[Tuple[int, ...]] = None,
    floats_strategy=default_floats_strategy,
) -> st.SearchStrategy[torch.Tensor]:
    """Strategy to generate PyTorch tensors."""
    array_strat = arrays_strategy(max_dims, max_size, shape, floats_strategy)
    return array_strat.map(torch.Tensor)
