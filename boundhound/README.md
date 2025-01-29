# 🐶 📈 BoundHound: A Branch-and-Bound Optimization Solver

BoundHound is an educational implementation of a [branch-and-bound](https://en.wikipedia.org/wiki/Branch_and_bound) solver for [Mixed Integer Programming (MIP)](https://en.wikipedia.org/wiki/Integer_programming) problems. While commercial solvers like [Gurobi](https://en.wikipedia.org/wiki/Gurobi), CPLEX, and CBC focus on performance, BoundHound emphasizes clarity to facilitate understanding of optimization algorithms.

## 1. Core Concepts

### 1.1 Linear Programming

[Linear programming](https://en.wikipedia.org/wiki/Linear_programming) forms the foundation of modern mathematical optimization. At its core, a linear program consists of a linear objective function to be optimized subject to linear constraints. The objective function takes the form c₁x₁ + c₂x₂ + ... + cₙxₙ, where each cᵢ represents the contribution of decision variable xᵢ to the objective value. The constraints similarly maintain linearity, expressed as linear combinations of variables bounded by constants: a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ ≤ b₁.

The fundamental property that makes linear programming tractable is the [convexity](https://en.wikipedia.org/wiki/Convex_optimization) of its feasible region. This geometric property ensures that any [local optimum](https://en.wikipedia.org/wiki/Local_optimum) is also globally optimal. The feasible region, formed by the intersection of the constraint [halfspaces](https://en.wikipedia.org/wiki/Half-space_(geometry)), creates a [polytope](https://en.wikipedia.org/wiki/Polytope) in n-dimensional space. This polytope's structure leads to one of the most important theoretical results in linear programming: the optimal solution, if it exists, occurs at a [vertex](https://en.wikipedia.org/wiki/Vertex_(geometry)) of this polytope.

Consider a production planning scenario where a manufacturer must decide production quantities for two products. Each product contributes to profit (the objective) while consuming limited resources (the constraints):

```
maximize  3x₁ + 2x₂             (profit from products 1 and 2)
subject to:
    2x₁ + x₂ ≤ 10              (resource 1 limitation)
    x₁ + 3x₂ ≤ 15              (resource 2 limitation)
    x₁, x₂ ≥ 0                 (non-negativity constraints)
```

This formulation embodies several key properties of linear programs. The feasible region is convex, meaning any line segment connecting two feasible points lies entirely within the feasible region. Vertices occur where constraint boundaries intersect, and in some cases, multiple constraints may intersect at a single point, leading to degeneracy—a crucial consideration in algorithmic implementation.

In BoundHound, linear programs are represented by the `LPProblem` class, which provides a flexible interface for problem construction:

```python
problem = LPProblem(
    c=np.array([3.0, 2.0]),             # Objective coefficients
    A_ub=np.array([[2.0, 1.0],          # Upper bound constraints
                   [1.0, 3.0]]),
    b_ub=np.array([10.0, 15.0]),        # Right-hand sides
    lb=np.array([0.0, 0.0])             # Non-negativity constraints
)
```

The class supports various constraint types:
- Upper and lower bound constraints (`A_ub`, `b_ub`, `A_lb`, `b_lb`)
- Equality constraints (`A_eq`, `b_eq`)
- Variable bounds (`lb`, `ub`)
- Objective sense (minimize/maximize)

### 1.2 Mixed Integer Programming

The extension to [mixed integer programming](https://en.wikipedia.org/wiki/Integer_programming) introduces a fundamental change in the mathematical structure of optimization problems. By requiring certain variables to take only integer values, we break the convexity property that makes linear programming tractable. The feasible region becomes a [discrete set](https://en.wikipedia.org/wiki/Discrete_set) of points, rather than a continuous polytope, fundamentally altering the nature of the optimization problem.

The introduction of integrality constraints transforms a [polynomial-time](https://en.wikipedia.org/wiki/Time_complexity#Polynomial_time) solvable problem into an [NP-hard](https://en.wikipedia.org/wiki/NP-hardness) one. This complexity arises from the [combinatorial](https://en.wikipedia.org/wiki/Combinatorial_optimization) nature of integer solutions—we can no longer rely on local improvement methods to find global optima. The number of potential integer solutions grows exponentially with problem size, and the loss of convexity means that local optima may not be globally optimal.

When we modify our production example to require integer quantities for the first product:
```
maximize  3x₁ + 2x₂
subject to:
    2x₁ + x₂ ≤ 10
    x₁ + 3x₂ ≤ 15
    x₁, x₂ ≥ 0
    x₁ must be integer
```

The problem becomes fundamentally harder. The optimal solution to the linear programming relaxation (ignoring integrality) provides an upper bound on the optimal integer solution, but the gap between these values—known as the integrality gap—can be substantial. This gap motivates many of the sophisticated techniques used in integer programming, including cutting planes that progressively tighten the relaxation and branching strategies that intelligently partition the solution space.

BoundHound implements mixed integer programs through the `MILPProblem` class, which combines an `LPProblem` with a set of integer variable indices:

```python
milp = MILPProblem(
    lp=problem,                    # Base linear program
    integer_vars={0}               # x₁ must be integer
)
```

The solution process, implemented in `BranchAndBoundSolver`, maintains a tree of nodes where each node represents a linear programming relaxation. The solver tracks:
- Global bounds through the `MILPSolution` class
- Node status (optimal, infeasible, unbounded)
- Solution quality via the `mip_gap` property
- Search tree visualization with the `render()` method

## 2. The Simplex Method

### 2.1 Standard Form and Transformations

The [simplex method](https://en.wikipedia.org/wiki/Simplex_algorithm) operates on a specific canonical form of linear programs, known as [standard form](https://en.wikipedia.org/wiki/Standard_form_(linear_programming)). This standardization transforms the diverse ways of expressing linear programs into a uniform structure.

The transformation process requires several carefully constructed steps. Inequality constraints must be converted to equations through the introduction of [slack variables](https://en.wikipedia.org/wiki/Slack_variable). For a "less than or equal to" constraint (≤), we add a non-negative slack variable that measures the unused capacity: 2x₁ + x₂ ≤ 10 becomes 2x₁ + x₂ + s₁ = 10, where s₁ ≥ 0. Conversely, "greater than or equal to" constraints (≥) require [surplus variables](https://en.wikipedia.org/wiki/Slack_variable) subtracted from the left side.

Variables with arbitrary sign (free variables) present another challenge. The standard form requires all variables to be non-negative, so we decompose each free variable into the difference of two non-negative variables: x = x⁺ - x⁻, where x⁺, x⁻ ≥ 0. This transformation doubles the number of variables but preserves the problem's structure.

The objective function must also be standardized. Maximization problems are converted to minimization by negating the objective coefficients, and any constant terms in the objective function can be moved outside the optimization problem, as they don't affect the optimal solution.

### 2.2 Basic Solutions and Pivoting

The simplex method's efficiency stems from its movement between [basic feasible solutions](https://en.wikipedia.org/wiki/Basic_feasible_solution)—points where exactly m variables (where m is the number of constraints) are non-zero, and these variables correspond to linearly independent columns of the constraint matrix A. These points are represented by a [basis matrix](https://en.wikipedia.org/wiki/Basis_(linear_algebra)) B, an m×m nonsingular submatrix of A, which allows us to compute the values of the basic variables as xᵦ = B⁻¹b.

The pivoting process that moves between basic feasible solutions involves sophisticated numerical computations. First, we compute reduced costs for non-basic variables using the formula c̄ₖ = cₖ - π^T aₖ, where π = c_B^T B⁻¹ represents the simplex multipliers. These reduced costs measure the rate of improvement in the objective function per unit increase in each non-basic variable.

The selection of entering and leaving variables requires careful consideration of numerical stability. While choosing the most negative reduced cost (in minimization) often works well, sophisticated implementations use more nuanced selection criteria. The ratio test, which determines how far we can move in the chosen direction, must account for degeneracy and numerical precision: min{xᵢ/aᵢₖ : aᵢₖ > 0}.

### 2.3 Revised Simplex Implementation

The [revised simplex method](https://en.wikipedia.org/wiki/Revised_simplex_method) represents a significant computational advancement over the original simplex algorithm. At its core lies the efficient maintenance of the basis inverse through [matrix factorization](https://en.wikipedia.org/wiki/Matrix_decomposition) techniques. Rather than working with the full inverse matrix B⁻¹, the method maintains an [LU factorization](https://en.wikipedia.org/wiki/LU_decomposition) B = LU, where L is lower triangular and U is upper triangular. This factorization allows for more stable [numerical computations](https://en.wikipedia.org/wiki/Numerical_analysis) and efficient updates as the basis changes.

The method employs sophisticated pricing strategies to reduce computational overhead. Instead of examining all non-basic variables at each iteration, partial pricing examines only a subset of promising candidates. This approach, while potentially requiring more iterations, often reduces overall computation time significantly. Multiple pricing extends this concept by selecting several entering variables simultaneously, allowing for parallel computation of steps. The steepest edge criterion provides a more nuanced approach to variable selection by considering the geometric interpretation of the simplex method, selecting moves that make the most progress toward optimality.

[Cycling](https://en.wikipedia.org/wiki/Cycling_(linear_programming)) prevention represents another crucial aspect of the implementation. While cycling (infinite loops between bases) is rare in practice, its possibility necessitates careful handling. [Bland's rule](https://en.wikipedia.org/wiki/Bland%27s_rule) provides a simple but effective anti-cycling mechanism by always selecting the eligible variable with the lowest index. More sophisticated implementations may employ perturbation methods, slightly modifying the problem data to avoid degenerate vertices, or use lexicographic ordering to ensure a unique path through the solution space.

### 2.4 Two-Phase Method

The two-phase method addresses the fundamental challenge of finding an initial feasible solution to a linear program. Phase I constructs an auxiliary problem by introducing artificial variables to create an obvious, though possibly non-optimal, starting point. These artificial variables are added to constraints where a feasible solution is not immediately apparent, particularly equality constraints and "greater than or equal to" inequalities.

The auxiliary objective function minimizes the sum of artificial variables, effectively attempting to drive them to zero. This phase serves a dual purpose: it determines whether the original problem has any feasible solutions while potentially constructing one if it exists. A positive minimum in the Phase I problem indicates infeasibility in the original problem—no solution exists that satisfies all constraints. Conversely, a zero minimum confirms feasibility and provides a valid starting basis for Phase II.

The transition between phases requires careful handling of the basis structure. When artificial variables remain in the basis at zero levels, they must be replaced through a series of pivot operations to maintain feasibility. The original objective function is then restored, but the feasible basis discovered in Phase I provides the crucial starting point for optimization.

## 3. Branch and Bound

The [branch-and-bound](https://en.wikipedia.org/wiki/Branch_and_bound) algorithm represents a sophisticated approach to mixed integer programming that combines the power of linear programming with intelligent enumeration. The method maintains a [tree structure](https://en.wikipedia.org/wiki/Tree_(data_structure)) where each node represents a linear programming relaxation with additional variable bounds. The root node contains the original problem with integer constraints relaxed, and subsequent nodes progressively restrict variables to integer values.

### 3.1 Tree Construction and Management

The efficiency of branch and bound heavily depends on the underlying data structures and management strategies. Each node in the tree maintains not only the problem data—constraints, bounds, and objective function—but also solution information and the history of branching decisions that led to its creation. This historical information proves invaluable for developing branching heuristics and warm-starting subsequent solves.

[Strong branching](https://en.wikipedia.org/wiki/Branch_and_bound#Variable_selection) represents a key technique for making intelligent branching decisions. By temporarily exploring the impact of different branching choices before committing to one, the algorithm can make more informed decisions at the cost of additional computation. Pseudo-costs track the historical impact of branching on particular variables, providing a computationally cheaper alternative to full strong branching while maintaining much of its effectiveness.

Memory management plays a crucial role in the implementation. The tree structure can grow exponentially, requiring careful balance between memory usage and computational efficiency. Node pools implement sophisticated storage strategies, potentially writing less promising nodes to disk while maintaining hot nodes in memory. Modern implementations often exploit parallel processing capabilities, exploring multiple branches simultaneously while ensuring proper synchronization of bounds and solutions.

### 3.2 Advanced Search Strategies

The search process through the branch-and-bound tree combines multiple sophisticated strategies. Best-bound search selects nodes with the most promising bound on the objective value, effectively minimizing the worst-case number of nodes needed to prove optimality. However, this approach can consume significant memory as the tree grows. Depth-first search, while potentially exploring more nodes, requires minimal memory and often finds feasible solutions quickly.

[Cutting planes](https://en.wikipedia.org/wiki/Cutting-plane_method) provide a powerful tool for strengthening the linear programming relaxations. [Gomory cuts](https://en.wikipedia.org/wiki/Cutting-plane_method#Gomory's_cut), derived from the simplex tableau, guarantee finite convergence in integer programming. [Lift-and-project cuts](https://en.wikipedia.org/wiki/Lift-and-project) exploit the structure of 0-1 variables to generate stronger inequalities. Cover inequalities identify and exploit subset relationships in knapsack-like constraints. The management of these cuts requires careful consideration—while they tighten the relaxation, too many cuts can slow down the linear programming solves.

[Primal heuristics](https://en.wikipedia.org/wiki/Heuristic_(computer_science)) complement the exact nature of branch and bound by quickly finding good feasible solutions. Simple rounding schemes provide fast but potentially low-quality solutions. The [feasibility pump](https://en.wikipedia.org/wiki/Feasibility_pump) alternates between finding integer solutions and optimizing continuous relaxations. [Local search](https://en.wikipedia.org/wiki/Local_search_(optimization)) methods explore the neighborhood of known solutions for improvements. Solution polishing attempts to improve existing feasible solutions through limited tree search or cutting plane generation.
