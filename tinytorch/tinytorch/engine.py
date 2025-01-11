"""
This module implements the core concepts of automatic differentiation (autodiff),
a technique for efficiently computing derivatives of functions defined by a
computational graph. It provides forward and reverse mode differentiation,
fundamental operations, and utilities for building and differentiating custom
functions. The implementation leverages numpy for efficient numerical operations.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from tinytorch.visualization import plot_graph

# Define float32 array type for clarity
Float32Array = NDArray[np.float32]

ArrayLike = Union[Float32Array, List[float | int], float, int]
TensorLike = Union[ArrayLike, "Tensor"]
AxisType = int | Tuple[int, ...]


class Operation(Enum):
    """Basic operations that are supported."""

    ADD = "+"
    MULT = "*"
    EXP = "exp"
    POW = "**"
    MAX = "max"
    MIN = "min"
    MATMUL = "@"
    SUM = "sum"
    STACK = "stack"
    IDENT = "ident"
    LOG = "log"


def _cast_array(data: ArrayLike) -> Float32Array:
    """Cast tensor-like object to numpy array with float32 dtype."""
    if isinstance(data, list):
        array = np.array(data, dtype=np.float32)
    elif isinstance(data, np.ndarray):
        array = data.astype(np.float32)
    elif isinstance(data, (int, float, np.float32, np.int32)):
        array = np.array(data, dtype=np.float32)
    else:
        raise TypeError("Wrong data type")
    # Handle scalar numpy arrays (0-dim arrays)
    if array.shape == () or array.ndim == 0:
        array = array.reshape(())
    return array


def _cast_tensor(x: TensorLike) -> Tensor:
    """Casts compatible datatypes to Tensor, raises if not compatible."""
    out = x if isinstance(x, Tensor) else Tensor(_cast_array(x))
    return out


def _sum_to_shape(x: Float32Array, shape: Tuple[int, ...]) -> Float32Array:
    """
    Sum array so that it matches 'shape', effectively reversing any broadcasting that expanded 'x' to a bigger shape.

    This version expands on the right (axis=-1), which matches how numpy broadcasting works.
    """
    # Case 1: If x has fewer dims, add trailing (rightmost) dims until ranks match.
    while x.ndim < len(shape):
        x = np.expand_dims(x, axis=-1)

    # Case 2: If x has more dims than shape, sum out the extra leading dims.
    while x.ndim > len(shape):
        x = x.sum(axis=0)

    # Case 3: Now x.ndim == len(shape). Walk from the right.
    for i in reversed(range(len(shape))):
        if shape[i] == 1 and x.shape[i] != 1:
            x = x.sum(axis=i, keepdims=True)

    return x


class Tensor(object):
    """Core tensor implementation."""

    def __init__(
        self,
        data: ArrayLike,
        label: Optional[str] = None,
        _children: Optional[Tuple[Tensor, ...]] = None,
        _op: Optional[Operation] = None,
    ) -> None:
        """Initialize a new Tensor with data and optional children and operation."""
        self.data: Float32Array = _cast_array(data)
        self.grad: Float32Array = np.zeros_like(data, dtype=np.float32)
        self.label = label
        self._op = _op
        self._children = set(_children) if _children is not None else set()
        self._backward = lambda: None

    def __repr__(self) -> str:
        """Return string representation of the tensor."""
        return (
            "Tensor"
            + (f"({self.label}, " if self.label is not None else "(")
            + f"shape={self.shape})"
        )

    def __str__(self) -> str:
        """Return string representation of the tensor."""
        return self.__repr__()

    def render(self, output_format: str = "png") -> None:
        """Renders the computational graph."""
        plot_graph(self, output_format)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor's data."""
        return self.data.shape

    def __len__(self) -> int:
        """Return the size of the first dimension of the tensor."""
        if self.data.shape == ():
            raise TypeError("len() of a 0-d tensor")
        return self.data.shape[0]

    def _broadcast_backward(self, grad_term: Float32Array) -> None:
        """Handle gradient accumulation for broadcasted operations."""
        if self.data.shape == ():  # Scalar special-case
            self.grad += grad_term.sum()
        else:
            grad_term = _sum_to_shape(grad_term, self.data.shape)
            self.grad += grad_term

    @staticmethod
    def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
        """Stack arrays along specified axis and new tensor with stacked data"""
        stacked = np.stack([t.data for t in tensors], axis=axis)
        out = Tensor(stacked, None, tuple(tensors), Operation.STACK)

        def _backward():
            """Split gradient along stacking axis and distribute to inputs."""
            grads = np.split(out.grad, len(tensors), axis=axis)
            for tensor, grad in zip(tensors, grads):
                # Squeeze out the stacking dimension before broadcasting
                grad_squeezed = np.squeeze(grad, axis=axis) if axis < grad.ndim else grad
                tensor._broadcast_backward(grad_squeezed)

        out._backward = _backward
        return out

    def __add__(self, other: TensorLike) -> Tensor:
        """Add other tensor-like object to this tensor.

        Forward pass: z = x + y
        Backward pass:
            dz/dx = 1
            dz/dy = 1
        """
        other = _cast_tensor(other)
        out = Tensor(self.data + other.data, None, (self, other), Operation.ADD)

        def _backward():
            """Local derivative is 1 for both inputs (gradients accumulate here)."""
            self._broadcast_backward(out.grad)
            other._broadcast_backward(out.grad)

        out._backward = _backward
        return out

    def __radd__(self, other: TensorLike) -> Tensor:
        """Handle addition when tensor is the right operand."""
        other = _cast_tensor(other)
        return self + other

    def __mul__(self, other: TensorLike) -> Tensor:
        """Multiply other tensor-like object with this tensor.

        Forward pass: z = x * y
        Backward pass:
            dz/dx = y  (derivative of x*y with respect to x is y)
            dz/dy = x  (derivative of x*y with respect to y is x)
        """
        other = _cast_tensor(other)
        out = Tensor(self.data * other.data, None, (self, other), Operation.MULT)

        def _backward():
            """Local derivatives are the opposite operand (gradients accumulate here)."""
            self._broadcast_backward(other.data * out.grad)
            other._broadcast_backward(self.data * out.grad)

        out._backward = _backward
        return out

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Handle multiplication when tensor is the right operand."""
        other = _cast_tensor(other)
        return self * other

    def __neg__(self) -> Tensor:
        """Return the negation of this tensor.

        Forward pass: z = -x
        Backward pass:
            dz/dx = -1  (derivative of -x with respect to x is -1)
        """
        out = Tensor(-self.data, None, (self,), Operation.MULT)

        def _backward():
            """Local derivative is -1."""
            self.grad += -1 * out.grad  # dL/dx = dL/dz * dz/dx = dL/dz * (-1)

        out._backward = _backward
        return out

    def __sub__(self, other: TensorLike) -> Tensor:
        """Subtract other tensor-like object from this tensor.

        Forward pass: z = x - y
        Backward pass:
            dz/dx = 1   (derivative of x-y with respect to x is 1)
            dz/dy = -1  (derivative of x-y with respect to y is -1)
        """
        other = _cast_tensor(other)
        out = Tensor(self.data - other.data, None, (self, other), Operation.ADD)

        def _backward():
            """Local derivatives are 1 and -1."""
            self._broadcast_backward(out.grad)
            other._broadcast_backward(-out.grad)

        out._backward = _backward
        return out

    def __rsub__(self, other: TensorLike) -> Tensor:
        """Handle subtraction when tensor is the right operand."""
        other = _cast_tensor(other)
        return other - self

    def __pow__(self, exponent: float | int) -> Tensor:
        """Raise tensor to the power of exponent.

        Forward pass: z = x^n
        Backward pass:
            dz/dx = n * x^(n-1)  (power rule)
        """
        out = Tensor(np.power(self.data, exponent), None, (self,), Operation.POW)

        def _backward():
            """Local derivative is n * x^(n-1)."""
            self._broadcast_backward((exponent * np.power(self.data, exponent - 1)) * out.grad)

        out._backward = _backward
        return out

    def __truediv__(self, other: TensorLike) -> Tensor:
        """Divide tensor by other tensor-like object."""
        other = _cast_tensor(other)
        out = self * (other**-1)
        return out

    def __rtruediv__(self, other: TensorLike) -> "Tensor":
        """Handle division when tensor is the denominator (other / self)."""
        other = _cast_tensor(other)
        return other / self

    def __matmul__(self, other: TensorLike) -> Tensor:
        """Matrix multiplication using @ operator.

        Forward pass: z = x @ y
        Backward pass:
            dz/dx = grad @ y.T
            dz/dy = x.T @ grad
        """
        other = _cast_tensor(other)
        out = Tensor(np.matmul(self.data, other.data), None, (self, other), Operation.MATMUL)

        def _backward():
            """Local derivatives using matrix multiplication rules."""
            self._broadcast_backward(np.matmul(out.grad, other.data.T))
            other._broadcast_backward(np.matmul(self.data.T, out.grad))

        out._backward = _backward
        return out

    def __rmatmul__(self, other: TensorLike) -> Tensor:
        """Handle matrix multiplication when tensor is the right operand."""
        other = _cast_tensor(other)
        return other @ self

    def exp(self) -> Tensor:
        """Compute element-wise exponential of tensor.

        Forward pass: z = e^x
        Backward pass:
            dz/dx = e^x  (derivative of e^x is itself)
        """
        out = Tensor(np.exp(self.data), None, (self,), Operation.EXP)

        def _backward():
            """Local derivative is e^x (the output itself)."""
            self._broadcast_backward(out.data * out.grad)

        out._backward = _backward
        return out

    def min(self, other: TensorLike) -> Tensor:
        """Return element-wise minimum between self and other.

        Forward pass: z = min(x, y)
        Backward pass:
            dz/dx = 1 if x < y else 0.5 if x == y else 0
            dz/dy = 1 if y < x else 0.5 if x == y else 0
        """
        other = _cast_tensor(other)
        out = Tensor(np.minimum(self.data, other.data), None, (self, other), Operation.MIN)

        def _backward():
            """Local derivative is 1 for the smaller input, 0.5 for equal inputs, 0 for larger."""
            equal = (self.data == other.data).astype(np.float32)
            less = (self.data < other.data).astype(np.float32)
            greater = (self.data > other.data).astype(np.float32)

            self._broadcast_backward(out.grad * (less + 0.5 * equal))
            other._broadcast_backward(out.grad * (greater + 0.5 * equal))

        out._backward = _backward
        return out

    def max(self, other: TensorLike) -> Tensor:
        """Return element-wise maximum between self and other.

        Forward pass: z = max(x, y)
        Backward pass:
            dz/dx = 1 if x > y else 0.5 if x == y else 0
            dz/dy = 1 if y > x else 0.5 if x == y else 0
        """
        other = _cast_tensor(other)
        out = Tensor(np.maximum(self.data, other.data), None, (self, other), Operation.MAX)

        def _backward():
            """Local derivative is 1 for the larger input, 0.5 for equal inputs, 0 for smaller."""
            equal = (self.data == other.data).astype(np.float32)
            greater = (self.data > other.data).astype(np.float32)
            less = (self.data < other.data).astype(np.float32)

            self._broadcast_backward(out.grad * (greater + 0.5 * equal))
            other._broadcast_backward(out.grad * (less + 0.5 * equal))

        out._backward = _backward
        return out

    def lin(self) -> Tensor:
        """Linear (identity) activation function.

        Forward pass: z = x
        Backward pass: dz/dx = 1
        """
        out = Tensor(self.data, None, (self,), Operation.IDENT)

        def _backward():
            """Local derivative is 1."""
            self._broadcast_backward(out.grad)

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """Compute hyperbolic tangent of tensor. For large values, we clip the input to avoid overflow in float32.

        Uses a numerically stable implementation:
            tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        """
        clipped = self.max(-100).min(100)
        exp_pos = clipped.exp()
        exp_neg = (-clipped).exp()
        out = (exp_pos - exp_neg) / (exp_pos + exp_neg)
        return out

    def sigmoid(self) -> Tensor:
        """Compute sigmoid activation function of tensor."""
        exp = (-1 * self).exp()
        num = 1
        den = 1 + exp
        out = num / den
        return out

    def relu(self) -> Tensor:
        """Compute ReLU activation function: max(0, x).

        Forward pass: z = max(0, x)
        Backward pass:
            dz/dx = 1 if x > 0 else 0
        """
        zero = _cast_tensor(0.0)
        out = self.max(zero)

        def _backward():
            """Local derivative is 1 where input was positive, 0 elsewhere."""
            self._broadcast_backward(out.grad * (self.data > 0).astype(np.float32))

        out._backward = _backward
        return out

    def sum(self, axis: Optional[AxisType] = None) -> Tensor:
        """Sum tensor over specified axes.

        Forward pass: z = sum(x, axis)
        Backward pass: Gradient is broadcasted back to match input shape
        """
        # Convert single axis to tuple for consistent handling
        if isinstance(axis, int):
            axis = (axis,)
        # If axis is None, sum over all axes
        if axis is None:
            axis = tuple(range(self.data.ndim))

        out = Tensor(np.sum(self.data, axis=axis), None, (self,), Operation.SUM)

        def _backward():
            """Local derivative is 1, broadcasted to match input shape."""
            self._broadcast_backward(out.grad)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        """Compute natural logarithm of tensor.

        Forward pass: z = ln(x)
        Backward pass:
            dz/dx = 1/x  (derivative of ln(x) is 1/x)
        """
        out = Tensor(np.log(self.data), None, (self,), Operation.LOG)

        def _backward():
            """Local derivative is 1/x."""
            self._broadcast_backward(out.grad / self.data)

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Compute gradients through back propagation.

        First builds a topologically sorted list of all nodes in the graph,
        starting from this tensor. Then iterates through the nodes in reverse
        order, calling _backward() on each to accumulate gradients.
        """
        topo: List[Tensor] = []
        visited = set()

        def _build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        # Go one tensor at a time and apply the chain rule.
        self.grad = np.ones_like(self.data, np.float32)  # Initialize with ones for root
        for tensor in reversed(topo):
            tensor._backward()
