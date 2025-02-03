# BoundHound 🐶 📈

BoundHound is an educational implementation of [branch-and-bound](https://en.wikipedia.org/wiki/Branch_and_bound) optimization in Python. While commercial solvers like Gurobi, CPLEX, and CBC focus on performance, BoundHound emphasizes clarity to facilitate understanding of optimization algorithms.

> **Educational Note**: This is a learning-focused implementation meant to illustrate the fundamentals of optimization algorithms. While functional, it is not intended for production use. For real applications, please use established solvers like Gurobi, CPLEX, or CBC.

## Features 🌟

BoundHound provides the fundamental building blocks needed for mixed-integer optimization:
- Linear Programming (LP) solver using the revised simplex method
- Mixed-Integer Linear Programming (MILP) solver using branch-and-bound
- Standard form transformations and problem preprocessing
- Efficient basis management and pivoting strategies
- Comprehensive property-based testing
- Interactive visualization of branch-and-bound trees

## Development Setup 🛠️

### Prerequisites

BoundHound is built with a minimal set of modern Python tools:
- [NumPy](https://numpy.org/) for efficient matrix operations
- [Ruff](https://github.com/astral-sh/ruff) for formatting and linting
- [MyPy](http://mypy-lang.org/) for static type checking
- [Pytest](https://docs.pytest.org/) for testing
- [Graphviz](https://graphviz.org/) for branch-and-bound tree visualizations

### Quick Start

```bash
# Clone the repository
git clone https://github.com/matzehuels/noodleworks.git
cd boundhound

# Create virtual environment and install dependencies
make setup

# Run tests to verify installation
make test

# Optional: Run type checking and linting
make check
make lint
```

## Core Concepts 🧠

### 1. Linear Programming Theory

[Linear programming](https://en.wikipedia.org/wiki/Linear_programming) forms the foundation of modern mathematical optimization. At its core, a linear program consists of:

1. A linear objective function: c^T x
2. Linear equality constraints: Ax = b
3. Non-negativity requirements: x ≥ 0

The fundamental property that makes linear programming tractable is the [convexity](https://en.wikipedia.org/wiki/Convex_optimization) of its feasible region. This geometric property ensures that any [local optimum](https://en.wikipedia.org/wiki/Local_optimum) is also globally optimal.

Consider a simple production problem:
```
maximize  3x₁ + 2x₂             (profit from products 1 and 2)
subject to:
    2x₁ + x₂ ≤ 10              (resource 1 limitation)
    x₁ + 3x₂ ≤ 15              (resource 2 limitation)
    x₁, x₂ ≥ 0                 (non-negativity constraints)
```

The feasible region, formed by the intersection of these constraints, creates a [polytope](https://en.wikipedia.org/wiki/Polytope). The optimal solution must occur at a vertex of this polytope - a fundamental theorem that underlies the simplex method.

### 2. Standard Form and Transformations

Before solving, all linear programs must be converted to standard form. This transformation process is more subtle than it might appear and requires careful handling of several cases:

#### Theory

A linear program in standard form has:
1. All constraints as equations (Ax = b)
2. All variables non-negative (x ≥ 0)
3. Minimization objective

The transformation process involves:

1. **Inequality Conversion**: Each inequality constraint must be converted to an equation by introducing slack or surplus variables:
   - For ≤: Add a non-negative slack (2x₁ + x₂ ≤ 10 → 2x₁ + x₂ + s₁ = 10)
   - For ≥: Subtract a non-negative surplus (2x₁ + x₂ ≥ 10 → 2x₁ + x₂ - s₁ = 10)

2. **Free Variable Handling**: Variables without sign restrictions must be split:
   - x = x⁺ - x⁻ where x⁺, x⁻ ≥ 0
   - This preserves linearity while ensuring non-negativity

3. **Objective Normalization**: 
   - Maximize cx → Minimize -cx
   - Constants in the objective don't affect optimization

#### Implementation

BoundHound implements these transformations in a specific order to maintain numerical stability:

```python
def to_standard_form(self) -> StandardForm:
    """Convert to standard form maintaining specific ordering:
    1. Variable bounds (x >= lb) in variable order
    2. Variable bounds (x <= ub) in variable order
    3. General lower bounds (Ax >= b)
    4. General upper bounds (Ax <= b)
    5. Equality constraints (Ax = b)
    """
```

### 3. The Two-Phase Simplex Method

The two-phase method solves a fundamental challenge: finding an initial feasible solution to start the simplex algorithm.

#### Theory

The challenge arises because the simplex method requires a starting basic feasible solution, but finding one for arbitrary constraints is as hard as solving the original LP. The two-phase method solves this by:

1. **Phase I**: Create and solve an auxiliary problem
   - Add artificial variables to make finding a feasible solution trivial
   - Minimize the sum of artificial variables
   - If minimum is 0, original problem is feasible
   - If minimum > 0, original problem is infeasible

2. **Phase II**: Solve the original problem
   - Use the feasible basis from Phase I
   - Remove artificial variables
   - Optimize original objective

Consider the system:
```
x₁ + 2x₂ = 4
x₁, x₂ ≥ 0
```

Phase I creates:
```
minimize    a₁              (artificial variable)
subject to:
    x₁ + 2x₂ + a₁ = 4      (original constraint + artificial)
    x₁, x₂, a₁ ≥ 0         (non-negativity)
```

If we can drive a₁ to zero, we've found a feasible solution to the original problem.

#### Implementation

BoundHound implements this through careful basis management:

```python
def solve_phase1(phase1_problem: StandardForm) -> tuple[bool, LPSolution]:
    """Find initial feasible solution by minimizing artificial variables.
    
    Returns:
        is_feasible: Whether original problem has feasible solution
        solution: Phase I solution with feasible basis (if found)
    """
    solution = RevisedSimplexSolver(problem=phase1_problem).solve()
    is_feasible = solution.status == SimplexStatus.OPTIMAL
    return is_feasible, solution

def to_phase2_form(self, phase1_solution: LPSolution) -> StandardForm:
    """Convert Phase I solution to Phase II problem.
    
    Strategy:
    1. Keep the same A matrix and b vector
    2. Restore original objective coefficients
    3. Keep artificial variables in matrix but with large positive
       coefficients to prevent them from re-entering the basis
    """
```

### 4. The Revised Simplex Method

The revised simplex method represents a significant computational advancement over the original simplex algorithm, focusing on efficient matrix operations and numerical stability.

#### Theory

The method operates on the concept of a basis - a set of m linearly independent columns from the constraint matrix A. Given a basis B, we can:
1. Compute the basic solution: xᵦ = B⁻¹b
2. Calculate reduced costs: c̄ₙ = cₙ - (cᵦᵀB⁻¹)Aₙ
3. Determine optimality: solution is optimal if c̄ₙ ≥ 0 for minimization

The key insight is maintaining the basis inverse B⁻¹ rather than the full tableau. This leads to three core operations:

1. **Computing Basic Solution**:
   ```
   xᵦ = B⁻¹b         (basic variables)
   xₙ = 0           (non-basic variables)
   ```

2. **Pricing (Finding Entering Variable)**:
   ```
   π = cᵦᵀB⁻¹       (simplex multipliers)
   c̄ₙ = cₙ - πᵀAₙ   (reduced costs)
   ```

3. **Ratio Test (Finding Leaving Variable)**:
   ```
   d = B⁻¹Aⱼ        (direction vector)
   θ = min{xᵢ/dᵢ : dᵢ > 0}  (maximum step length)
   ```

#### Implementation

BoundHound implements these operations with careful attention to numerical stability:

```python
class RevisedSimplexSolver:
    """Revised simplex method solver for linear programs in standard form."""
    
    @property
    def basic_solution(self) -> NDArray[T]:
        """Current basic solution x_B = B^{-1}b."""
        result = self._basis_inverse @ self.problem.b
        return np.array(result, dtype=self.problem.b.dtype)
    
    @property
    def reduced_costs(self) -> NDArray[T]:
        """Reduced costs c̄ = c - πᵀA."""
        pi = self.simplex_multipliers
        result = self.problem.c - (pi @ self.problem.A)
        return np.array(result, dtype=self.problem.c.dtype)
    
    @property
    def leaving_variable(self) -> int | None:
        """Select leaving variable using minimum ratio test."""
        d = self._basis_inverse @ self.problem.A[:, self.entering_variable]
        ratios = np.full_like(d, np.inf)
        positive_d = d > self.tol
        ratios[positive_d] = self.basic_solution[positive_d] / d[positive_d]
        return int(np.argmin(ratios))
```

### 5. Mixed Integer Programming

Mixed Integer Programming (MIP) extends linear programming by requiring some variables to take integer values. This seemingly simple change fundamentally alters the problem's complexity.

#### Theory

The key challenge in MIP comes from the loss of convexity. Consider:
```
maximize    5x₁ + 4x₂
subject to:
    x₁ + x₂ ≤ 3.5
    x₁, x₂ ≥ 0
    x₁, x₂ integer
```

The LP relaxation (ignoring integrality) has solution (2.5, 1), but the integer optimal is (2, 1). This illustrates two key concepts:

1. **LP Relaxation**: Solving the problem without integer constraints
   - Provides an upper bound for maximization
   - Faster to solve than the integer problem
   - May have fractional values

2. **Integrality Gap**: Difference between integer and LP optimal values
   - Measures problem difficulty
   - Guides branching decisions
   - Used in optimality proofs

#### Implementation

BoundHound represents MIPs through a combination of LP and integer requirements:

```python
@dataclass
class MILPProblem:
    """A mixed-integer linear programming problem."""
    lp: LPProblem
    integer_vars: set[int]  # Indices of integer variables

    def is_integer_feasible(self, x: NDArray[np.float64]) -> bool:
        """Check if solution satisfies integrality constraints."""
        return all(abs(x[i] - round(x[i])) <= self.tol 
                  for i in self.integer_vars)
```

### 6. Branch and Bound

Branch and bound provides a systematic way to solve MIPs by dividing the problem into smaller subproblems and using bounds to prune unpromising branches. The algorithm combines the power of linear programming relaxations with intelligent enumeration.

#### Theory

The algorithm maintains a tree where each node represents an LP with additional variable bounds. The fundamental idea is to partition the feasible region into smaller subregions that can be either solved directly or proven to not contain the optimal solution.

Consider a MIP with optimal value z*. At any point in the algorithm:
- The best integer solution found so far (incumbent) provides a lower bound L ≤ z*
- The best LP relaxation among open nodes provides an upper bound U ≥ z*
- The optimality gap is (U - L)/|L|, measuring solution quality

Three key operations guide the search:

1. **Node Selection**: Choose which subproblem to solve next
   - Best-bound: Select node with highest upper bound (maximization)
   - Depth-first: Select deepest node to find feasible solutions quickly
   - Best-estimate: Combine bound and estimate of integer solution quality

2. **Branching**: When a solution x* has fractional values:
   - Most-fractional: Select variable xᵢ furthest from integrality
   - Strong branching: Evaluate multiple candidates
   - Create two child problems:
     * P₁: Original problem + (xᵢ ≤ ⌊x*ᵢ⌋)
     * P₂: Original problem + (xᵢ ≥ ⌈x*ᵢ⌉)

3. **Bounding and Pruning**: Use LP relaxations to eliminate subproblems
   - By bound: If node's LP value ≤ incumbent (maximization)
   - By infeasibility: If node's LP is infeasible
   - By optimality: If found integer solution at node

For example, consider maximizing x₁ + 2x₂ subject to:
```
x₁ + x₂ ≤ 2.5
x₁, x₂ ≥ 0
x₁, x₂ integer
```

The root LP relaxation gives (x₁, x₂) = (0, 2.5). Since x₂ is fractional:
1. Branch on x₂ ≤ 2 and x₂ ≥ 3
2. Right branch (x₂ ≥ 3) is infeasible → prune
3. Left branch (x₂ ≤ 2) gives (0.5, 2)
4. Branch on x₁ ≤ 0 and x₁ ≥ 1
5. Best integer solution: (0, 2) with value 4

#### Implementation

BoundHound implements branch and bound through careful node and tree management. Each node tracks:
- The LP relaxation with current bounds
- Parent and depth in search tree
- Solution status and value
- Branching decisions made
- Best integer solution in subtree

The main solution loop:
1. Initialize with root node (LP relaxation)
2. While nodes remain and not hit limit:
   a. Select most promising node (best bound)
   b. Solve node's LP relaxation
   c. If integer feasible:
      - Update incumbent if better
      - Prune by optimality
   d. If fractional:
      - Select branching variable
      - Create and queue child nodes
   e. Update global bounds
   f. Prune nodes that cannot improve incumbent
3. Return best solution found with optimality certificate

The solver maintains important state information:
- Active nodes queue for processing
- Global upper/lower bounds
- Best integer solution (incumbent)
- Search tree for visualization
- Node processing history

This information enables:
1. Solution quality certificates (optimality gap)
2. Search tree visualization
3. Performance analysis
4. Early termination with bounds

## Example Usage

For complete, step-by-step examples of solving optimization problems, check out our detailed Jupyter notebooks:

1. [Facility Location](examples/facility_location.ipynb) - An in-depth look at:
   - Mixed-integer programming
   - Branch and bound visualization
   - Solution strategies
   - Performance analysis