# TinyTorch 🔥

TinyTorch is a minimal implementation of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) in Python, designed to demonstrate the core concepts behind modern deep learning frameworks like PyTorch. By focusing on essential functionality, it provides clear insights into how automatic differentiation works in practice. This repo was greatly inspired by [micrograd](https://github.com/karpathy/micrograd), and extends its functionality from scalars to [tensors](https://en.wikipedia.org/wiki/Tensor).

> **Educational Note**: This is a learning-focused implementation meant to illustrate the fundamentals of deep learning frameworks. While functional, it is not intended for production use. For real applications, please use established frameworks like PyTorch or TensorFlow.

## Features 🌟

TinyTorch provides the fundamental building blocks needed for automatic differentiation:
- Tensor operations (`+, -, *, /, **, @, sum, min, max, log`) with automatic gradient tracking
- Basic neural network activation functions
- Broadcasting support for scalar operations
- Type safety through annotations and strict checking
- Comprehensive [property-based testing](https://en.wikipedia.org/wiki/Property_testing)
- Collection of examples for building neural networks

## Development Setup 🛠️

### Prerequisites

TinyTorch is built with a minimal set of modern Python tools:
- [NumPy](https://numpy.org/) for efficient array operations
- [Ruff](https://github.com/astral-sh/ruff) for formatting and linting
- [MyPy](http://mypy-lang.org/) for static type checking
- [Pytest](https://docs.pytest.org/) and [Hypothesis](https://hypothesis.works/) for robust testing
- [PyTorch](https://pytorch.org/) for gradient verification
- [Graphviz](https://graphviz.org/) for graph visualizations

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/tinytorch.git
cd tinytorch

# Create virtual environment and install dependencies
make setup

# Run tests to verify installation
make test

# Optional: Run type checking and linting
make check
make lint
```

## Core Concepts 🧠

### 1. Automatic Differentiation

Automatic differentiation (AD) is distinct from both [symbolic differentiation](https://en.wikipedia.org/wiki/Computer_algebra) and [numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation). While symbolic differentiation manipulates mathematical expressions and numerical differentiation uses finite differences, AD computes derivatives by decomposing expressions into elementary operations and applying the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) systematically.

#### Theory

AD has two primary modes:
- **Forward Mode**: Computes derivatives alongside values during forward evaluation (efficient for few inputs, many outputs)
- **Reverse Mode**: Records operations during forward pass, then computes derivatives backward (efficient for many inputs, few outputs)

The reverse mode is particularly efficient for neural networks where we typically compute gradients of a scalar loss with respect to many parameters. For a deeper understanding, see [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767).

#### The Chain Rule in Practice

Consider a simple computation: `f(x) = sin(x²)`. We can break this into elementary operations:
1. `g(x) = x²`
2. `f(x) = sin(g(x))`

The chain rule tells us:
```
df/dx = df/dg * dg/dx
      = cos(g(x)) * 2x
      = cos(x²) * 2x
```

This simple example demonstrates how complex derivatives can be computed by combining simpler ones. TinyTorch extends this concept to handle multivariate functions and complex computational graphs automatically.

## Implementation Deep Dive 🔍

### 1. Computational Graphs and Automatic Differentiation

A [computational graph](https://en.wikipedia.org/wiki/Data-flow_programming) is a directed acyclic graph (DAG) that represents the computation. Each node is an operation, and edges show how data flows between operations. In TinyTorch, we build this graph dynamically during the forward pass by tracking parent-child relationships between operations.

#### Graph Construction

Every `Tensor` object maintains a set of its immediate dependencies (children) and a function that knows how to compute gradients with respect to these dependencies. Here's how it works:

```python
class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        label: str | None = None,
        _children: tuple[Tensor, ...] | None = None,
        _op: Operation | None = None,
    ) -> None:
        """Initialize a new Tensor with data and optional children and operation."""
        self.data = _cast_array(data)
        self.grad = np.zeros_like(data, dtype=np.float32)
        self.label = label
        self._op = _op
        self._children = set(_children) if _children is not None else set()
        self._backward = lambda: None
```

When we perform operations, we create new tensors that remember their inputs:

```python
def __mul__(self, other):
    """Multiply operation (z = x * y)."""
    other = _cast_tensor(other)
    # Create new tensor with operation inputs as children
    out = Tensor(self.data * other.data, _children=(self, other))
    
    def _backward():
        # Gradient computation using the chain rule
        self.grad += other.data * out.grad    # ∂z/∂x = y * ∂L/∂z
        other.grad += self.data * out.grad    # ∂z/∂y = x * ∂L/∂z
    
    out._backward = _backward
    return out
```

#### Topological Sorting for Backpropagation

During backpropagation, we need to process nodes in the correct order: from the output back to the inputs. This requires a [topological sort](https://en.wikipedia.org/wiki/Topological_sorting) of the computational graph. We use depth-first search to build this ordering:

```python
def backward(self):
    # Build topologically sorted list of all nodes
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    # Go one variable at a time and apply the chain rule
    self.grad = np.ones_like(self.data)  # Initialize gradient at output
    for v in reversed(topo):             # Reverse order: from outputs to inputs
        v._backward()                    # Apply this node's portion of the chain rule
```

#### The Multivariate Chain Rule and Broadcasting

In neural networks, we often work with tensors of different shapes that get broadcast together. This complicates gradient computation because we need to:
1. Apply the [multivariate chain rule](https://en.wikipedia.org/wiki/Chain_rule#Higher_dimensions)
2. Handle gradient accumulation across broadcasted dimensions

Consider matrix multiplication `C = A @ B`:
```python
def __matmul__(self, other):
    """Matrix multiplication (C = A @ B)."""
    other = _cast_tensor(other)
    out = Tensor(np.matmul(self.data, other.data), _children=(self, other))
    
    def _backward():
        # For C = A @ B:
        # ∂L/∂A = ∂L/∂C @ B.T
        # ∂L/∂B = A.T @ ∂L/∂C
        self.grad += np.matmul(out.grad, other.data.T)
        other.grad += np.matmul(self.data.T, out.grad)
    
    out._backward = _backward
    return out
```

Broadcasting adds another layer of complexity. When tensors are broadcast together, we need to sum gradients across the broadcasted dimensions during backpropagation:

```python
def _broadcast_backward(self, grad_term: NDArray[np.float32]) -> None:
    """Handle gradient accumulation for broadcasted operations."""
    if self.data.shape == ():  # Scalar special-case
        self.grad += grad_term.sum()
    else:
        grad_term = _sum_to_shape(grad_term, self.data.shape)
        self.grad += grad_term
```

This careful handling of gradients ensures that operations like:
```python
x = Tensor([1, 2, 3])           # Shape: (3,)
y = Tensor([[1], [2]])          # Shape: (2,1)
z = x + y                       # Shape: (2,3) via broadcasting
```
correctly propagate gradients back through the computation graph.

### 2. Neural Network Architecture

TinyTorch's neural network implementation follows a modular design inspired by both biological neural networks and modern deep learning frameworks. The architecture is built around three core abstractions: neurons, layers, and networks. Let's explore how these components work together and how to use them effectively.

#### Building Neural Networks with TinyTorch

At the heart of TinyTorch's neural network API is the `Module` base class, which provides a foundation for all neural network components. The key components can be constructed as follows:

```python
# Create a single layer
layer = Layer(n_input=2, n_neurons=3, activation=Activation.RELU)

# Create a multi-layer network
mlp = MLP(
    n_input=2,                           # Input dimension
    layers=[
        (64, Activation.RELU),           # Hidden layer: 64 neurons, ReLU
        (32, Activation.RELU),           # Hidden layer: 32 neurons, ReLU
        (1, Activation.SIGMOID),         # Output layer: 1 neuron, Sigmoid
    ]
)
```

#### Neuron: The Fundamental Building Block

The artificial neuron is the basic computational unit in neural networks. In TinyTorch, a neuron is implemented as a specialized `Module` that performs a weighted sum of its inputs followed by a nonlinear activation:

```python
# Single neuron with 2 inputs and ReLU activation
neuron = Neuron(n_input=2, activation=Activation.RELU)

# Forward pass through a neuron
x = Tensor([1.0, 2.0])
y = neuron(x)  # Computes activation(w⋅x + b)
```

Each neuron maintains:
- A weight vector `w` of shape `(1, n_input)`
- A bias scalar `b`
- An activation function

The mathematical computation performed by a neuron is:
```python
z = w⋅x + b        # Weighted sum (dot product)
y = activation(z)  # Nonlinear transformation
```

During initialization, weights are set using Xavier/Glorot initialization:
```python
scale = np.sqrt(2.0 / (n_input + 1))  # Scale factor
w = np.random.normal(0, scale, size=(1, n_input))  # Weight initialization
b = np.zeros((1,))  # Bias initialization
```

This initialization scheme helps maintain stable gradients by keeping the variance of activations and gradients consistent across layers.

#### Layer: Parallel Processing Units

A layer combines multiple neurons operating in parallel. Each layer performs an affine transformation followed by a nonlinear activation:

```python
layer(x) = activation(W⋅x + b)
```

where:
- W is the weight matrix (n_neurons × n_input)
- b is the bias vector (n_neurons)
- activation is a nonlinear function like ReLU or Sigmoid

The weights are initialized using Xavier/Glorot initialization to maintain stable gradients throughout the network:

```
W ~ Normal(0, sqrt(2/(n_input + n_neurons)))
```

#### Forward and Backward Flow

When you pass data through the network, it flows through each layer sequentially:

```python
# Forward pass
x = Tensor([[1.0, 2.0]])  # Input features
y = mlp(x)                # Predictions

# Backward pass (after computing loss)
loss.backward()           # Compute gradients
```

During the forward pass, each layer computes:
1. Affine transformation: z = Wx + b
2. Activation: a = activation(z)

During the backward pass, gradients flow in reverse, using the chain rule:
```
∂L/∂W = ∂L/∂a * ∂a/∂z * ∂z/∂W
∂L/∂b = ∂L/∂a * ∂a/∂z * ∂z/∂b
```

#### Training Neural Networks

Training a neural network involves repeatedly:
1. Making predictions
2. Computing loss
3. Updating parameters

Here's how this looks in practice:

```python
# Define model and loss
model = MLP(n_input=2, layers=[(64, Activation.RELU), (1, Activation.SIGMOID)])
learning_rate = 0.01

# Training loop
for epoch in range(n_epochs):
    # Forward pass
    predictions = model(X_train)
    loss = binary_cross_entropy(predictions, y_train)
    
    # Backward pass
    model.flush_grads()  # Zero gradients
    loss.backward()      # Compute gradients
    
    # Update parameters
    for param in model.parameters:
        param.data -= learning_rate * param.grad
```

The beauty of automatic differentiation means you don't need to manually derive or implement these gradients - TinyTorch handles this automatically through its computational graph.

#### Advanced Features

While TinyTorch focuses on fundamental architectures, its modular design supports experimentation with advanced concepts:

- Custom activation functions by subclassing `Module`
- Layer composition for complex architectures
- Flexible loss functions and optimizers
- Batch processing for efficient training

The Universal Approximation Theorem guarantees that even this simple architecture can approximate any continuous function, given sufficient width. In practice, the depth of the network (number of layers) often proves more important for learning hierarchical representations of data.

### 3. Activation Functions

Activation functions introduce nonlinearity into neural networks, allowing them to learn complex patterns. TinyTorch implements several common activations:

#### ReLU (Rectified Linear Unit)
```python
def relu(self) -> Tensor:
    """ReLU activation: max(0, x)
    
    Forward: y = max(0, x)
    Backward: dy/dx = 1 if x > 0 else 0
    """
    out = Tensor(np.maximum(0, self.data), _children=(self,))
    
    def _backward():
        self.grad += (out.data > 0) * out.grad
    
    out._backward = _backward
    return out
```

#### Sigmoid
```python
def sigmoid(self) -> Tensor:
    """Sigmoid activation: 1 / (1 + exp(-x))
    
    Forward: y = 1 / (1 + exp(-x))
    Backward: dy/dx = y * (1 - y)
    """
    y = 1 / (1 + np.exp(-self.data))
    out = Tensor(y, _children=(self,))
    
    def _backward():
        self.grad += y * (1 - y) * out.grad
    
    out._backward = _backward
    return out
```

These activations are carefully chosen because:
- ReLU prevents the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
- Sigmoid squashes outputs to [0,1], perfect for binary classification
- Their derivatives are simple and computationally efficient

### 4. Training Examples

TinyTorch provides a collection of examples that demonstrate how to use the framework for various machine learning tasks. These examples are designed to be educational, showing both the mathematical concepts and their practical implementation.

#### Basic Operations

Here are some fundamental operations that form the building blocks of neural networks:

```python
# Creating tensors
x = tt.Tensor([1, 2, 3])
y = tt.Tensor([4, 5, 6])

# Basic arithmetic with broadcasting
z = x + y  # Addition: elementwise [5, 7, 9]
z = x * y  # Element-wise multiplication: [4, 10, 18]
z = x @ y  # Matrix multiplication (dot product): 32

# Common neural network operations
z = x.relu()    # ReLU activation: [1, 2, 3]
z = x.sigmoid() # Sigmoid activation: [0.731, 0.881, 0.953]

# Gradient computation
z = (x * y).sum()  # Forward pass
z.backward()       # Backward pass
print(x.grad)      # Shows ∂z/∂x = y
print(y.grad)      # Shows ∂z/∂y = x
```

#### End-to-End Examples

For complete, step-by-step examples of building and training neural networks, check out our detailed Jupyter notebooks:

1. [Binary Classification](examples/classifier.ipynb) - A comprehensive guide to:
   - Preparing and visualizing data
   - Building a multi-layer perceptron
   - Training with gradient descent
   - Monitoring model performance
   - Visualizing decision boundaries

This notebook provides extensive explanations and visualizations to help you understand how TinyTorch works in practice.
