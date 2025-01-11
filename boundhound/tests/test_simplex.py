"""Tests for tableau-based simplex solver implementation.

Tests verify:
1. Basic LP solving (maximization, minimization)
2. Two-phase method handling
3. Edge cases (infeasible, unbounded)
4. Numerical stability
"""

import numpy as np

from boundhound.core import LPProblem
from boundhound.simplex import solve_lp
from boundhound.types import Sense, SimplexStatus


def test_simple_maximization():
    """Test simple maximization problem:
    maximize    2x + 3y
    subject to  x + y ≤ 4
                x ≤ 3
                x, y ≥ 0

    The optimal solution should be x=0, y=4 with objective value 12.
    """
    problem = LPProblem(
        c=np.array([2.0, 3.0]),
        A_ub=np.array([[1.0, 1.0], [1.0, 0.0]]),
        b_ub=np.array([4.0, 3.0]),
        lb=np.zeros(2),
        sense=Sense.MAX,
    )

    # Solve using simplex
    solution = solve_lp(problem)

    # Check solution
    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 0.0)  # x = 0
    np.testing.assert_almost_equal(solution.x[1], 4.0)  # y = 4
    np.testing.assert_almost_equal(solution.value, 12.0)  # 2*0 + 3*4 = 12


def test_maximization_with_equality():
    """Test a maximization problem with equality constraints:

    maximize    3x + 2y
    subject to  2x + y = 18
                2x + 3y <= 42
                x, y >= 0

    The optimal solution is x=3, y=12 with z=33.
    """
    problem = LPProblem(
        c=np.array([3.0, 2.0]),
        A_eq=np.array([[2.0, 1.0]]),
        b_eq=np.array([18.0]),
        A_ub=np.array([[2.0, 3.0]]),
        b_ub=np.array([42.0]),
        lb=np.zeros(2),
        sense=Sense.MAX,
    )

    solution = solve_lp(problem)

    # Check solution
    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 3.0)  # x = 3
    np.testing.assert_almost_equal(solution.x[1], 12.0)  # y = 12
    np.testing.assert_almost_equal(solution.value, 33.0)  # 3*3 + 2*12 = 33


def test_simple_minimization():
    """Test simple minimization problem:
    minimize    -x + 2y
    subject to  x + y >= 3
                x <= 4
                y >= 2
                x, y >= 0

    The optimal solution should be x=1, y=2 with objective value 3.
    """
    problem = LPProblem(
        c=np.array([-1.0, 2.0]),
        A_lb=np.array([[1.0, 1.0]]),
        b_lb=np.array([3.0]),
        A_ub=np.array([[1.0, 0.0]]),
        b_ub=np.array([4.0]),
        lb=np.array([0.0, 2.0]),
        sense=Sense.MIN,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 4.0)  # x = 4
    np.testing.assert_almost_equal(solution.x[1], 2.0)  # y = 2
    np.testing.assert_almost_equal(solution.value, 0.0)  # -1*4 + 2*2 = 0


def test_maximization_with_mixed_bounds():
    """Test maximization with mixed bounds:
    maximize    2x + 3y
    subject to  x + y >= 3
                x <= 4
                y <= 5        # Add upper bound on y to make problem bounded
                y >= 2
                x, y >= 0

    The optimal solution should be x=4, y=5 with objective value 23:
    - y's coefficient (3) is larger than x's (2), so maximize y first
    - y is bounded by y <= 5, so y = 5
    - x's coefficient is positive, so maximize x subject to x <= 4
    - Therefore x = 4
    - This satisfies x + y >= 3 since 4 + 5 = 9 >= 3
    - Objective value = 2(4) + 3(5) = 8 + 15 = 23
    """
    problem = LPProblem(
        c=np.array([2.0, 3.0]),
        A_lb=np.array([[1.0, 1.0]]),  # x + y >= 3
        b_lb=np.array([3.0]),
        A_ub=np.array(
            [
                [1.0, 0.0],  # x <= 4
                [0.0, 1.0],
            ]
        ),  # y <= 5
        b_ub=np.array([4.0, 5.0]),
        lb=np.array([0.0, 2.0]),  # x,y >= 0 and y >= 2
        sense=Sense.MAX,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 4.0)  # x = 4
    np.testing.assert_almost_equal(solution.x[1], 5.0)  # y = 5
    np.testing.assert_almost_equal(solution.value, 23.0)  # 2*4 + 3*5 = 23


def test_infeasible_problem():
    """Test infeasible linear program:
    maximize    x + y
    subject to  x + y <= 1
                x + y >= 2
                x, y >= 0

    This is clearly infeasible since x + y cannot be both <= 1 and >= 2.
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([1.0]),
        A_lb=np.array([[1.0, 1.0]]),
        b_lb=np.array([2.0]),
        lb=np.array([0.0, 0.0]),
        sense=Sense.MAX,
    )

    solution = solve_lp(problem)
    assert solution.status == SimplexStatus.INFEASIBLE
    assert solution.x is None
    assert solution.value is None


def test_unbounded_problem():
    """Test unbounded linear program:
    maximize    x + y
    subject to  -x + y <= 1
                x, y >= 0

    This is unbounded because we can increase x arbitrarily while keeping
    y = x + 1 to satisfy the constraint, making the objective x + (x + 1)
    grow without bound.
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        A_ub=np.array([[-1.0, 1.0]]),
        b_ub=np.array([1.0]),
        lb=np.array([0.0, 0.0]),
        sense=Sense.MAX,
    )

    solution = solve_lp(problem)
    assert solution.status == SimplexStatus.UNBOUNDED
    assert solution.x is None
    assert solution.value is None


def test_minimization_with_equality():
    """Test minimization with equality constraints:
    minimize    x + 2y
    subject to  x + y = 5
                x + 2y >= 8
                x, y >= 0

    The optimal solution should be x=2, y=3 with objective value 8:
    - x + y = 5 forces us to stay on that line
    - x + 2y >= 8 requires y >= (8-x)/2
    - Given these, minimum occurs at x=2, y=3
    """
    problem = LPProblem(
        c=np.array([1.0, 2.0]),
        A_eq=np.array([[1.0, 1.0]]),
        b_eq=np.array([5.0]),
        A_lb=np.array([[1.0, 2.0]]),
        b_lb=np.array([8.0]),
        lb=np.zeros(2),
        sense=Sense.MIN,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 2.0)  # x = 2
    np.testing.assert_almost_equal(solution.x[1], 3.0)  # y = 3
    np.testing.assert_almost_equal(solution.value, 8.0)  # 1*2 + 2*3 = 8


def test_maximization_with_lower_bounds():
    """Test maximization with only lower bound constraints:
    maximize    2x + y
    subject to  x + y >= 4
                x >= 2
                y >= 1
                x, y >= 0

    The optimal solution should be unbounded since we can increase x
    indefinitely while keeping y=1 to satisfy x + y >= 4.
    """
    problem = LPProblem(
        c=np.array([2.0, 1.0]),
        A_lb=np.array([[1.0, 1.0]]),
        b_lb=np.array([4.0]),
        lb=np.array([2.0, 1.0]),
        sense=Sense.MAX,
    )

    solution = solve_lp(problem)
    assert solution.status == SimplexStatus.UNBOUNDED
    assert solution.x is None
    assert solution.value is None


def test_minimization_with_upper_bounds():
    """Test minimization with only upper bound constraints:
    minimize    -x - 2y
    subject to  x + y <= 5
                x <= 3
                y <= 4
                x, y >= 0

    The optimal solution should be x=1, y=4 with objective value -9:
    - Since we're minimizing -x - 2y, this is equivalent to maximizing x + 2y
    - y has larger coefficient (2 vs 1), so maximize y first
    - y is bounded by min(4, 5) = 4
    - With y=4, x + 4 <= 5 implies x <= 1
    - Therefore x=1 is optimal
    """
    problem = LPProblem(
        c=np.array([-1.0, -2.0]),
        A_ub=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        b_ub=np.array([5.0, 3.0, 4.0]),
        lb=np.zeros(2),
        sense=Sense.MIN,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 1.0)  # x = 1
    np.testing.assert_almost_equal(solution.x[1], 4.0)  # y = 4
    np.testing.assert_almost_equal(solution.value, -9.0)  # -1*1 + -2*4 = -9


def test_maximization_all_constraint_types():
    """Test maximization with all types of constraints:
    maximize    3x + 2y
    subject to  x + y = 6      (equality)
                2x + y >= 8    (lower bound)
                x + 2y <= 14   (upper bound)
                x >= 2         (variable lower bound)
                y <= 5         (variable upper bound)
                x, y >= 0

    The optimal solution should be x=6, y=0 with objective value 18:
    - x + y = 6 forces y = 6-x
    - Since coefficient of x (3) is larger than y (2), solver maximizes x
    - x + y = 6 and x + 2y <= 14 allow x=6, y=0
    - This satisfies 2x + y >= 8 since 12 > 8
    - x >= 2 and y <= 5 are satisfied
    """
    problem = LPProblem(
        c=np.array([3.0, 2.0]),
        A_eq=np.array([[1.0, 1.0]]),
        b_eq=np.array([6.0]),
        A_lb=np.array([[2.0, 1.0]]),
        b_lb=np.array([8.0]),
        A_ub=np.array([[1.0, 2.0]]),
        b_ub=np.array([14.0]),
        lb=np.array([2.0, 0.0]),
        ub=np.array([np.inf, 5.0]),
        sense=Sense.MAX,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 6.0)  # x = 6
    np.testing.assert_almost_equal(solution.x[1], 0.0)  # y = 0
    np.testing.assert_almost_equal(solution.value, 18.0)  # 3*6 + 2*0 = 18


def test_fractional_solution():
    """Test problem with integer coefficients that has fractional solution:
    maximize    5x + 4y
    subject to  x + y <= 4
                2x + y <= 7
                x, y >= 0

    The optimal solution should be x=3, y=1 with objective value 19:
    - Solving graphically, the feasible region is bounded by:
      * x + y = 4 (line 1)
      * 2x + y = 7 (line 2)
      * x = 0, y = 0 (non-negativity)
    - The optimal point is at intersection of lines 1 and 2
    - Solving: 2x + y = 7 and x + y = 4 gives x = 3, y = 1
    """
    problem = LPProblem(
        c=np.array([5.0, 4.0]),
        A_ub=np.array([[1.0, 1.0], [2.0, 1.0]]),
        b_ub=np.array([4.0, 7.0]),
        lb=np.zeros(2),
        sense=Sense.MAX,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 3.0)  # x = 3
    np.testing.assert_almost_equal(solution.x[1], 1.0)  # y = 1
    np.testing.assert_almost_equal(solution.value, 19.0)  # 5*3 + 4*1 = 19


def test_redundant_constraints():
    """Test problem with redundant constraints:
    minimize    x + y
    subject to  x + y >= 3       (binding)
                2x + 2y >= 6     (redundant, same as first constraint)
                x + 2y >= 4      (binding)
                x, y >= 0

    The optimal solution should be x=2, y=1 with objective value 3:
    - The second constraint is redundant (multiply first by 2)
    - Solving the binding constraints:
      * x + y = 3
      * x + 2y = 4
    - Subtracting equations: -y = -1, so y = 1
    - Substituting back: x + 1 = 3, so x = 2
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        A_lb=np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]]),
        b_lb=np.array([3.0, 6.0, 4.0]),
        lb=np.zeros(2),
        sense=Sense.MIN,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 2.0)  # x = 2
    np.testing.assert_almost_equal(solution.x[1], 1.0)  # y = 1
    np.testing.assert_almost_equal(solution.value, 3.0)  # 1*2 + 1*1 = 3


def test_multiple_optimal_solutions():
    """Test problem with multiple optimal solutions:
    maximize    x + y
    subject to  x + y <= 4
                x <= 3
                y <= 3
                x, y >= 0

    This problem has multiple optimal solutions:
    - The objective x + y = 4 is parallel to the constraint x + y <= 4
    - Any point on the line x + y = 4 that satisfies other constraints is optimal
    - The optimal value is 4, achieved by:
      * x = 1, y = 3
      * x = 2, y = 2
      * x = 3, y = 1
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        A_ub=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        b_ub=np.array([4.0, 3.0, 3.0]),
        lb=np.zeros(2),
        sense=Sense.MAX,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    # The solver should find one of the optimal solutions
    x, y = solution.x[0], solution.x[1]
    np.testing.assert_almost_equal(x + y, 4.0)  # Sum should be 4
    assert 0 <= x <= 3 and 0 <= y <= 3  # Within bounds
    np.testing.assert_almost_equal(solution.value, 4.0)  # Optimal value is 4


def test_cycling_prevention():
    """Test problem that could cycle without anti-cycling rules:
    minimize    -x - y
    subject to   x + y <= 3
                -x + 2y <= 2
                 2x - y <= 4
                 x, y >= 0

    This is a classic example where cycling could occur without
    anti-cycling rules (like Bland's rule). The problem has multiple
    optimal solutions with objective value -3, including:
    - (x=2, y=1)
    - (x=4/3, y=5/3)
    Both solutions:
    - Satisfy all constraints
    - Achieve objective value -3
    - Lie at intersections of constraints
    """
    problem = LPProblem(
        c=np.array([-1.0, -1.0]),
        A_ub=np.array([[1.0, 1.0], [-1.0, 2.0], [2.0, -1.0]]),
        b_ub=np.array([3.0, 2.0, 4.0]),
        lb=np.zeros(2),
        sense=Sense.MIN,
    )

    solution = solve_lp(problem)

    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    # Check that we found an optimal solution
    x, y = solution.x[0], solution.x[1]
    # Verify constraints
    np.testing.assert_almost_equal(x + y, 3.0)  # First constraint should be binding
    assert -x + 2 * y <= 2.0  # Second constraint
    assert 2 * x - y <= 4.0  # Third constraint
    assert x >= 0 and y >= 0  # Non-negativity
    np.testing.assert_almost_equal(solution.value, -3.0)  # Optimal value is -3


def test_simple_lp():
    """Test simple LP problem.

    max 3x + 4y
    s.t. x + 2y <= 14  (upper bound)
         3x - y <= 0   (upper bound)
         x, y >= 0

    The optimal solution is x=2, y=6 with objective value 30:
    - From 3x - y <= 0, we get y >= 3x
    - Substituting into x + 2y <= 14: x + 2(3x) <= 14
    - This simplifies to 7x <= 14, so x <= 2
    - With x=2, y >= 6 from the first constraint
    - But y=6 is optimal since we want to maximize 4y
    """
    problem = LPProblem(
        c=np.array([3, 4]),
        A_ub=np.array([[1, 2], [3, -1]]),  # Convert to upper bound form
        b_ub=np.array([14, 0]),
        sense=Sense.MAX,
    )
    solution = solve_lp(problem)
    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_almost_equal(solution.x[0], 2.0)  # x = 2
    np.testing.assert_almost_equal(solution.x[1], 6.0)  # y = 6
    np.testing.assert_almost_equal(solution.value, 30.0)  # 3*2 + 4*6 = 30


def test_infeasible_lp():
    """Test infeasible LP problem.

    max x + y
    s.t. x + y <= 1  (upper bound)
         x + y >= 2  (lower bound)
         x, y >= 0

    This is clearly infeasible since x + y cannot be both <= 1 and >= 2.
    """
    problem = LPProblem(
        c=np.array([1, 1]),
        A_ub=np.array([[1, 1]]),  # x + y <= 1
        b_ub=np.array([1]),
        A_lb=np.array([[1, 1]]),  # x + y >= 2
        b_lb=np.array([2]),
        sense=Sense.MAX,
    )
    solution = solve_lp(problem)
    assert solution.status == SimplexStatus.INFEASIBLE
    assert solution.x is None
    assert solution.value is None


def test_unbounded_lp():
    """Test unbounded LP problem.

    max x + y
    s.t. x - y <= 0  (upper bound)
         x, y >= 0

    This is unbounded because we can increase y arbitrarily while keeping x = y,
    making the objective x + y grow without bound.
    """
    problem = LPProblem(
        c=np.array([1, 1]),
        A_ub=np.array([[1, -1]]),  # x - y <= 0
        b_ub=np.array([0]),
        sense=Sense.MAX,
    )
    solution = solve_lp(problem)
    assert solution.status == SimplexStatus.UNBOUNDED
    assert solution.x is None
    assert solution.value is None


def test_degenerate_lp():
    """Test degenerate LP problem.

    min -x - y
    s.t. x + y <= 1   (upper bound)
         x - y <= 0   (upper bound)
         -x + y <= 0  (upper bound)
         x, y >= 0

    The optimal solution is x = y = 0.5 with objective value -1.
    This problem is degenerate because multiple constraints are active at the optimum.
    """
    problem = LPProblem(
        c=np.array([-1, -1]),
        A_ub=np.array([[1, 1], [1, -1], [-1, 1]]),
        b_ub=np.array([1, 0, 0]),
        sense=Sense.MIN,
    )
    solution = solve_lp(problem)
    assert solution.status == SimplexStatus.OPTIMAL
    assert solution.x is not None
    assert abs(solution.value + 1) <= 1e-6  # Objective value should be -1
    assert abs(solution.x[0] - 0.5) <= 1e-6  # x = 0.5
    assert abs(solution.x[1] - 0.5) <= 1e-6  # y = 0.5
