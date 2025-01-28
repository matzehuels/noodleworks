"""Module for graph visualizations and other plots."""

from typing import TYPE_CHECKING

import numpy as np
from graphviz import Digraph

if TYPE_CHECKING:
    from tinytorch.engine import Tensor

MAX_ELEMENTS = 3


def format_array(arr: np.ndarray) -> str:
    """Format numpy array for display."""
    # For single elements, just show the value
    if arr.size == 1:
        return f"{float(arr):.3f}"

    # For small arrays (<=3 elements), show all values
    if arr.size <= MAX_ELEMENTS:
        return np.array2string(
            arr,
            precision=3,
            suppress_small=True,
            separator=",",
            floatmode="fixed",
        ).replace("\n", "")

    # For larger arrays, show shape only
    return f"shape={arr.shape}"


def plot_graph(t: "Tensor", output_format: str = "png") -> None:
    """Plot computational graph for a tensor using graphviz."""
    dot = Digraph(t.label or "Tensor", format=output_format)
    dot.attr(rankdir="LR")  # Left to right direction
    visited: dict[Tensor, str] = {}

    def _trace(tensor: "Tensor") -> str:
        if tensor in visited:
            return visited[tensor]

        tensor_key = tensor.label or f"t_{len(visited)}"
        visited[tensor] = tensor_key

        # More compact node format
        dot.node(
            tensor_key,
            f"{{{tensor_key} | {format_array(tensor.data)}}}",
            shape="record",
        )

        # If this is a leaf tensor (no operation/children), we're done
        if tensor._op is None or tensor._children is None:
            return tensor_key

        # Add operation node
        op_key = f"{tensor_key}_op"
        dot.node(op_key, tensor._op.value, shape="circle")

        # Connect operation to result (current tensor)
        dot.edge(op_key, tensor_key)

        # Recursively process children and connect them to the operation
        for child in tensor._children:
            child_key = _trace(child)
            dot.edge(child_key, op_key)

        return tensor_key

    # Start the recursive trace from the root tensor
    _trace(t)

    # Render the graph
    dot.render("computational_graph", view=True, cleanup=True)
