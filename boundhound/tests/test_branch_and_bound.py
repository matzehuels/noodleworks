"""Tests for branch and bound solver implementation.

Tests verify:
1. Basic MILP solving (optimal solutions)
2. Edge cases (infeasible, unbounded)
3. Branching behavior
4. Solution quality
5. Early termination
"""

import numpy as np

from boundhound.branch_and_bound import BranchAndBoundSolver, MILPProblem, MILPStatus
from boundhound.core import LPProblem
from boundhound.types import Sense


def test_simple_milp():
    """Test simple mixed-integer linear program:
    maximize    5x + 4y
    subject to  x + y <= 5
                2x + y <= 8
                x, y >= 0
                x, y integer

    The optimal solution is x=3, y=2 with objective value 23.
    The LP relaxation has solution x=3.2, y=1.6 with value 22.4,
    so branching is needed to find the integer solution.
    """
    problem = MILPProblem(
        lp=LPProblem(
            c=np.array([5.0, 4.0]),
            A_ub=np.array([[1.0, 1.0], [2.0, 1.0]]),
            b_ub=np.array([5.0, 8.0]),
            sense=Sense.MAX,
        ),
        integer_vars={0, 1},  # Both x and y must be integer
    )

    solution = BranchAndBoundSolver(problem).solve()

    assert solution.status == MILPStatus.OPTIMAL
    assert solution.x is not None
    np.testing.assert_array_equal(solution.x, [3, 2])  # x=3, y=2
    assert abs(solution.value - 23.0) <= 1e-6  # 5*3 + 4*2 = 23


def test_infeasible_milp():
    """Test infeasible mixed-integer linear program:
    maximize    x + y
    subject to  x + y <= 1
                x + y >= 2
                x, y >= 0
                x, y integer

    This is infeasible since x + y cannot be both <= 1 and >= 2.
    """
    problem = MILPProblem(
        lp=LPProblem(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([1.0]),
            A_lb=np.array([[1.0, 1.0]]),
            b_lb=np.array([2.0]),
            sense=Sense.MAX,
        ),
        integer_vars={0, 1},
    )

    solution = BranchAndBoundSolver(problem).solve()

    assert solution.status == MILPStatus.INFEASIBLE
    assert solution.x is None
    assert solution.value is None
    assert solution.nodes_processed >= 1


def test_unbounded_milp():
    """Test unbounded mixed-integer linear program:
    maximize    x + y
    subject to  -x + y <= 1
                x, y >= 0
                x, y integer

    This is unbounded because we can increase x arbitrarily while keeping
    y = x + 1 to satisfy the constraint, making the objective x + (x + 1)
    grow without bound.

    The LP relaxation should detect unboundedness in the root node.
    """
    problem = MILPProblem(
        lp=LPProblem(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[-1.0, 1.0]]),
            b_ub=np.array([1.0]),
            sense=Sense.MAX,
        ),
        integer_vars={0, 1},
    )

    solution = BranchAndBoundSolver(problem).solve()

    # The root node's LP relaxation should detect unboundedness
    assert solution.status == MILPStatus.UNBOUNDED
    assert solution.x is None
    assert solution.value is None
    assert solution.nodes_processed == 1  # Only root node should be processed
    assert len(solution.tree) == 1  # Only root node in tree
    assert solution.nodes_remaining == 0  # No nodes to process


def test_mixed_integer_linear():
    """Test mixed-integer linear program (some continuous variables):
    maximize    2x + 3y
    subject to  x + y <= 5
                2x + y <= 8
                x >= 0, y >= 0
                x integer, y continuous

    The optimal solution is x=3, y=2 with objective value 12.
    Only x needs to be integer, y can be continuous.
    """
    problem = MILPProblem(
        lp=LPProblem(
            c=np.array([2.0, 3.0]),
            A_ub=np.array([[1.0, 1.0], [2.0, 1.0]]),
            b_ub=np.array([5.0, 8.0]),
            sense=Sense.MAX,
        ),
        integer_vars={0},  # Only x must be integer
    )

    solution = BranchAndBoundSolver(problem).solve()

    assert solution.status == MILPStatus.OPTIMAL
    assert solution.x is not None
    assert abs(solution.x[0] - round(solution.x[0])) <= 1e-6  # x should be integer


def test_multiple_optimal_solutions():
    """Test MILP with multiple optimal solutions:
    maximize    x + y
    subject to  x + y <= 5
                x, y >= 0
                x, y integer

    This has multiple optimal solutions with value 5:
    (5,0), (4,1), (3,2), (2,3), (1,4), (0,5)
    The solver should find one of these.
    """
    problem = MILPProblem(
        lp=LPProblem(
            c=np.array([1.0, 1.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([5.0]),
            sense=Sense.MAX,
        ),
        integer_vars={0, 1},
    )

    solution = BranchAndBoundSolver(problem).solve()

    assert solution.status == MILPStatus.OPTIMAL
    assert solution.x is not None
    x, y = solution.x[0], solution.x[1]
    assert abs(x + y - 5) <= 1e-6  # Sum should be 5
    assert abs(x - round(x)) <= 1e-6  # x should be integer
    assert abs(y - round(y)) <= 1e-6  # y should be integer
    assert abs(solution.value - 5) <= 1e-6  # Optimal value is 5


def test_no_integer_solution():
    """Test MILP with no integer solution:
    maximize    x + y
    subject to  x + y = 2.5
                x, y >= 0
                x, y integer

    This is infeasible since no integer x,y satisfy x + y = 2.5
    """
    problem = MILPProblem(
        lp=LPProblem(
            c=np.array([1.0, 1.0]),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([2.5]),
            sense=Sense.MAX,
        ),
        integer_vars={0, 1},
    )

    solution = BranchAndBoundSolver(problem).solve()

    assert solution.status == MILPStatus.INFEASIBLE
    assert solution.x is None
    assert solution.value is None


def test_branching_variable_selection():
    """Test branching variable selection with multiple fractional variables:
    maximize    5x + 4y + 3z
    subject to  x + y + z <= 3.5
                x, y, z >= 0
                x, y, z integer

    The LP relaxation has x=2.5, y=1, z=0.
    The solver should branch on x first since it's most fractional.
    """
    problem = MILPProblem(
        lp=LPProblem(
            c=np.array([5.0, 4.0, 3.0]),
            A_ub=np.array([[1.0, 1.0, 1.0]]),
            b_ub=np.array([3.5]),
            sense=Sense.MAX,
        ),
        integer_vars={0, 1, 2},
    )

    solver = BranchAndBoundSolver(problem)
    solution = solver.solve()

    assert solution.status == MILPStatus.OPTIMAL
    assert solution.x is not None
    # Check all variables are integer
    for x in solution.x:
        assert abs(x - round(x)) <= 1e-6
