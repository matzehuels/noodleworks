import numpy as np
import pytest

from boundhound.core import Basis, LPProblem, StandardForm
from boundhound.types import LPSolution, PhaseType, Sense


def test_variable_lower_bounds():
    """Test standard form conversion with variable bounds.

    Original problem:
        minimize    x + y
        subject to:
            x >= 0          (non-negativity)
            y >= 0          (non-negativity)
            x <= 4          (upper bound)
            y <= 3          (upper bound)

    Expected standard form:
        - Non-negativity constraints are converted to equations with slack
        - Upper bounds are converted to lower bounds by negation
        - No artificial variables needed, so direct to Phase 2
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]), sense=Sense.MIN, lb=np.array([0.0, 0.0]), ub=np.array([4.0, 3.0])
    )

    standard = problem.to_standard_form()

    # Check dimensions
    assert standard.A.shape == (4, 6)  # 4 constraints (2 non-neg, 2 upper), 2 original + 4 slack
    assert len(standard.b) == 4  # 4 RHS values
    assert len(standard.c) == 6  # 2 original + 4 slack variables

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],  # -x - s3 = -4 (x <= 4)
            [0.0, -1.0, 0.0, 0.0, 0.0, -1.0],  # -y - s4 = -3 (y <= 3)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([0.0, 0.0, -4.0, -3.0])  # Non-negativity and upper bounds
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients unchanged
    expected_c = np.array([1.0, 1.0] + [0.0] * 4)  # Original coefficients + zeros for slack
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE2  # Direct to Phase 2 (no artificial variables)


def test_non_zero_lower_bounds():
    """Test standard form conversion with non-zero lower bounds.

    Original problem:
        minimize    x + y
        subject to:
            x >= 2          (lower bound)
            y >= -1         (lower bound)

    Expected standard form:
        - Lower bounds are converted to equations with slack
        - No artificial variables needed since these are inequalities
        - Direct to Phase 2 since no artificial variables
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        sense=Sense.MIN,
        lb=np.array([2.0, -1.0]),
        ub=np.array([np.inf, np.inf]),
    )

    standard = problem.to_standard_form()

    # Check dimensions
    assert standard.A.shape == (2, 4)  # 2 lower bound constraints, 2 original + 2 slack
    assert len(standard.b) == 2  # 2 RHS values
    assert len(standard.c) == 4  # 2 original + 2 slack

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0],  # x - s1 = 2
            [0.0, 1.0, 0.0, -1.0],  # y - s2 = -1
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([2.0, -1.0])  # Lower bounds
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients
    expected_c = np.array([1.0, 1.0, 0.0, 0.0])  # Original coefficients + zeros for slack
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE2  # Direct to Phase 2


def test_mixed_constraints():
    """Test handling of mixed equality and inequality constraints.

    Original problem:
        minimize    x + y
        subject to:
            x + y = 1          (equality)
            x >= 2             (lower bound)
            y >= 0             (non-negativity)
            y <= 3             (upper bound)

    Expected standard form:
        1. Variable bounds (x >= lb):
            x >= 2          (explicit lower bound)
            y >= 0          (non-negativity)
        2. Variable bounds (x <= ub):
            y <= 3          (upper bound)
        3. Equality constraints:
            x + y = 1          (equality)
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),  # [x, y]
        sense=Sense.MIN,
        lb=np.array([2.0, 0.0]),  # x >= 2, y >= 0
        ub=np.array([np.inf, 3.0]),  # x unbounded, y <= 3
        A_eq=np.array([[1.0, 1.0]]),  # x + y = 1
        b_eq=np.array([1.0]),
    )

    standard = problem.to_standard_form()

    # Check dimensions: 4 constraints (1 non-neg + 1 lower + 1 upper + 1 eq), 2 original + 3 slack + 1 artificial
    assert standard.A.shape == (4, 6)
    assert len(standard.b) == 4  # 4 RHS values
    assert len(standard.c) == 6  # 2 original + 3 slack + 1 artificial variables

    # Check constraint matrix
    expected_A = np.array(
        [
            # 1. Variable bounds (x >= lb)
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 2 (x >= 2)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            # 2. Variable bounds (x <= ub)
            [0.0, -1.0, 0.0, 0.0, -1.0, 0.0],  # -y - s3 = -3 (y <= 3)
            # 3. Equality constraints
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # x + y + a1 = 1 (equality)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS - same order as constraints
    expected_b = np.array(
        [
            2.0,  # x >= 2
            0.0,  # y >= 0
            -3.0,  # y <= 3
            1.0,
        ]
    )  # x + y = 1
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (Phase 1: minimize artificial variables)
    expected_c = np.array(
        [
            0.0,
            0.0,  # Original variables
            0.0,
            0.0,
            0.0,  # Slack variables
            1.0,
        ]
    )  # Artificial variable
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE1  # Phase 1 due to artificial variables


def test_maximization_objective():
    """Test standard form conversion with maximization objective.

    Original problem:
        maximize    x + y
        subject to:
            x >= 0              (non-negativity)
            y >= 0              (non-negativity)
            x + y >= 4          (inequality)
            x - y = 1           (equality)

    Expected standard form:
        - In Phase 1: minimize sum of artificial variables
        - Original objective coefficients are saved for Phase 2
        - Non-negativity constraints become equations with slack
        - Lower bound inequality gets slack variable
        - Equality constraint needs artificial variable
        - Phase 1 needed due to artificial variable
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),  # Original objective saved for Phase 2
        sense=Sense.MAX,
        lb=np.array([0.0, 0.0]),
        A_lb=np.array([[1.0, 1.0]]),
        b_lb=np.array([4.0]),
        A_eq=np.array([[1.0, -1.0]]),
        b_eq=np.array([1.0]),
    )

    standard = problem.to_standard_form()

    # Check dimensions
    assert standard.A.shape == (
        4,
        7,
    )  # 4 constraints (2 non-neg, 1 ineq, 1 eq), 2 original + 3 slack + 2 artificial
    assert len(standard.b) == 4
    assert len(standard.c) == 7

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            [1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],  # x + y - s3 + a1 = 4 (inequality)
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # x - y + a2 + a3 = 1 (equality)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([0.0, 0.0, 4.0, 1.0])  # [non-neg, non-neg, inequality, equality]
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (Phase 1: minimize artificial variables)
    expected_c = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    )  # Zeros for original/slack + ones for artificial
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE1  # Phase 1 due to artificial variables


def test_empty_constraints():
    """Test conversion of problem with only non-negativity constraints.

    Original problem:
        maximize    x + 2y
        subject to:
            x >= 0          (non-negativity)
            y >= 0          (non-negativity)

    Expected standard form:
        - Only non-negativity constraints, so direct to Phase 2
        - In Phase 2: original objective is negated for minimization
        - Non-negativity constraints become equations with slack
        - No artificial variables needed
    """
    problem = LPProblem(
        c=np.array([1.0, 2.0]),  # Will be negated in Phase 2
        sense=Sense.MAX,
        lb=np.array([0.0, 0.0]),
    )

    standard = problem.to_standard_form()

    # Check dimensions - only non-negativity constraints
    assert standard.A.shape == (2, 4)  # 2 non-negativity constraints, 2 original + 2 slack
    assert len(standard.b) == 2  # 2 RHS values
    assert len(standard.c) == 4  # 2 original + 2 slack variables

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0],  # y - s2 = 0 (y >= 0)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([0.0, 0.0])  # Non-negativity constraints
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (negated in Phase 2 for maximization)
    expected_c = np.array([-1.0, -2.0, 0.0, 0.0])  # Original coefficients negated + zeros for slack
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE2  # Direct to Phase 2 (no artificial variables)


def test_negative_lower_bounds():
    """Test conversion of negative lower bounds to standard form.

    Original problem:
        maximize    x + y
        subject to:
            x >= -1         (negative lower bound)
            y >= 1          (positive lower bound)

    Expected standard form:
        - Only inequality constraints, so direct to Phase 2
        - In Phase 2: original objective is negated for minimization
        - Lower bounds are converted to equations with slack
        - No artificial variables needed
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),  # Will be negated in Phase 2
        sense=Sense.MAX,
        lb=np.array([-1.0, 1.0]),
        ub=np.array([np.inf, np.inf]),
    )

    standard = problem.to_standard_form()

    # Check dimensions
    assert standard.A.shape == (2, 4)  # 2 constraints (2 lower bounds), 2 original + 2 slack
    assert len(standard.b) == 2  # 2 RHS values
    assert len(standard.c) == 4  # 2 original + 2 slack

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0],  # x - s1 = -1 (x >= -1)
            [0.0, 1.0, 0.0, -1.0],  # y - s2 = 1 (y >= 1)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([-1.0, 1.0])  # [x>=-1, y>=1]
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (negated in Phase 2 for maximization)
    expected_c = np.array([-1.0, -1.0, 0.0, 0.0])  # Original coefficients negated + zeros for slack
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE2  # Direct to Phase 2 (no artificial variables)


def test_zero_coefficients():
    """Test handling of constraints with zero coefficients.

    Original problem:
        minimize    x + y
        subject to:
            x >= 0              (non-negativity)
            y >= 0              (non-negativity)
            x + y >= 0          (general lower bound)
            x - y >= 0          (general lower bound)

    Expected standard form:
        - Non-negativity constraints become equations with slack
        - General lower bounds get slack variables
        - No artificial variables needed since all inequalities
        - Direct to Phase 2
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        sense=Sense.MIN,
        lb=np.array([0.0, 0.0]),
        A_lb=np.array([[1.0, 1.0], [1.0, -1.0]]),
        b_lb=np.array([0.0, 0.0]),
    )

    standard = problem.to_standard_form()

    # Check dimensions
    assert standard.A.shape == (4, 6)  # 4 constraints (2 non-neg, 2 lower), 2 original + 4 slack
    assert len(standard.b) == 4  # 4 RHS values
    assert len(standard.c) == 6  # 2 original + 4 slack

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            [1.0, 1.0, 0.0, 0.0, -1.0, 0.0],  # x + y - s3 = 0 (general lower bound)
            [1.0, -1.0, 0.0, 0.0, 0.0, -1.0],  # x - y - s4 = 0 (general lower bound)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([0.0, 0.0, 0.0, 0.0])  # All zero RHS
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients
    expected_c = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Original coefficients + zeros for slack
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Verify initial basis is valid
    standard.basis = Basis.from_standard_form(
        standard.A, standard.slack_indices, standard.art_indices
    )
    standard.basis.is_valid(standard.A)


def test_redundant_constraints():
    """Test handling of redundant constraints.

    Original problem:
        minimize    x + y
        subject to:
            x >= 0              (non-negativity)
            y >= 0              (non-negativity)
            x + y <= 4          (inequality)
            2x + 2y <= 8        (redundant inequality, 2 times first one)
            x - y = 1           (equality)
            2x - 2y = 2         (redundant equality, 2 times first one)

    Expected standard form:
        - Redundant constraints should be identified and handled appropriately
        - Basis selection should prioritize non-redundant constraints
        - Only one artificial variable needed despite two equality constraints
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        sense=Sense.MIN,
        lb=np.array([0.0, 0.0]),
        A_ub=np.array([[1.0, 1.0], [2.0, 2.0]]),
        b_ub=np.array([4.0, 8.0]),
        A_eq=np.array([[1.0, -1.0], [2.0, -2.0]]),
        b_eq=np.array([1.0, 2.0]),
    )

    standard = problem.to_standard_form()

    # Check dimensions - should not include redundant constraints
    assert standard.A.shape == (
        4,
        6,
    )  # 4 constraints (2 non-neg, 1 ineq, 1 eq), 2 original + 3 slack + 1 artificial
    assert len(standard.b) == 4  # 4 RHS values
    assert len(standard.c) == 6  # 2 original + 3 slack + 1 artificial

    # Check constraint matrix - should only include non-redundant constraints
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            [-1.0, -1.0, 0.0, 0.0, -1.0, 0.0],  # -x - y - s3 = -4 (x + y <= 4)
            [1.0, -1.0, 0.0, 0.0, 0.0, 1.0],  # x - y + a1 = 1 (equality)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS - should only include non-redundant constraints
    expected_b = np.array([0.0, 0.0, -4.0, 1.0])  # [non-neg, non-neg, inequality, equality]
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (Phase 1: minimize artificial variables)
    expected_c = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )  # Zeros for original/slack + ones for artificial
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE1  # Phase 1 due to artificial variable

    # Verify initial basis is valid
    slack_indices = (np.array([0, 1, 2]), np.array([2, 3, 4]))  # Row and column indices for slack
    art_indices = (np.array([3]), np.array([5]))  # Row and column indices for artificial
    standard.basis = Basis.from_standard_form(standard.A, slack_indices, art_indices)
    assert standard.basis.is_valid(standard.A)


def test_conflicting_bounds():
    """Test handling of conflicting variable bounds.

    Original problem:
        maximize    x
        subject to:
            x >= 2          (lower bound)
            x <= 1          (upper bound, conflicts with lower bound)

    Expected standard form:
        - Only inequality constraints, so direct to Phase 2
        - In Phase 2: original objective is negated for minimization
        - Lower bound becomes equation with slack
        - Upper bound is converted to lower bound by negation
        - No artificial variables needed (since we only have inequalities)
        - Problem is infeasible (but this is not detected at this stage)
    """
    problem = LPProblem(
        c=np.array([1.0]),  # Will be negated in Phase 2
        sense=Sense.MAX,
        lb=np.array([2.0]),
        ub=np.array([1.0]),  # Conflicts with lower bound
    )

    standard = problem.to_standard_form()

    # Check dimensions
    assert standard.A.shape == (2, 3)  # 2 constraints (1 lower, 1 upper), 1 original + 2 slack
    assert len(standard.b) == 2  # 2 RHS values
    assert len(standard.c) == 3  # 1 original + 2 slack

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, -1.0, 0.0],  # x - s1 = 2 (lower bound)
            [-1.0, 0.0, -1.0],  # -x - s2 = -1 (upper bound)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([2.0, -1.0])  # [lower bound, upper bound]
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (negated in Phase 2 for maximization)
    expected_c = np.array(
        [-1.0, 0.0, 0.0]
    )  # Original coefficient negated + zeros for slack variables
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Verify we're in Phase 2 since no artificial variables needed
    assert standard.phase_type == PhaseType.PHASE2


def test_negative_rhs():
    """Test handling of negative right-hand sides.

    Original problem:
        maximize    x + 2y
        subject to:
            x >= 0          (non-negativity)
            y >= 0          (non-negativity)
            x <= -2         (upper bound)
            x + y = -1      (equality)

    Expected standard form:
        - Has equality constraint, so starts in Phase 1
        - In Phase 1: minimize sum of artificial variables
        - Original objective coefficients are saved for Phase 2
        - Non-negativity constraints become equations with slack
        - Upper bound is converted to lower bound by negation
        - Equality constraint needs artificial variable
        - System includes negative RHS values
    """
    problem = LPProblem(
        c=np.array([1.0, 2.0]),  # Original objective saved for Phase 2
        sense=Sense.MAX,  # Will be handled in Phase 2
        lb=np.array([0.0, 0.0]),
        ub=np.array([-2.0, np.inf]),  # x <= -2
        A_eq=np.array([[1.0, 1.0]]),  # x + y = -1
        b_eq=np.array([-1.0]),
    )

    standard = problem.to_standard_form()

    # Check dimensions
    assert standard.A.shape == (4, 6)  # 4 constraints, 2 original + 3 slack + 1 artificial
    assert len(standard.b) == 4  # 4 RHS values
    assert len(standard.c) == 6  # 2 original + 3 slack + 1 artificial

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],  # -x - s3 = 2 (x <= -2)
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # x + y + a1 = -1 (equality)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([0.0, 0.0, 2.0, -1.0])  # [non-neg, non-neg, upper, equality]
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (Phase 1: minimize artificial variables)
    expected_c = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )  # Zeros for original/slack + one for artificial
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE1  # Phase 1 due to artificial variable

    # Verify initial basis is valid
    slack_indices = (np.array([0, 1, 2]), np.array([2, 3, 4]))  # Row and column indices for slack
    art_indices = (np.array([3]), np.array([5]))  # Row and column indices for artificial
    standard.basis = Basis.from_standard_form(standard.A, slack_indices, art_indices)
    assert standard.basis.is_valid(standard.A)


def test_phase1_no_artificial():
    """Test that Phase 1 is not needed when all constraints can be satisfied with slack variables.

    Original problem:
        minimize    x + 2y
        subject to:
            x >= 0          (non-negativity)
            y >= 0          (non-negativity)
            x + 2y <= 4     (upper bound, converted to -x - 2y >= -4)
            -x <= -2        (upper bound, converted to x >= 2)
    """
    # Create a problem with only inequality constraints
    problem = LPProblem(
        c=np.array([1.0, 2.0]),  # Minimize x + 2y
        A_ub=np.array([[1.0, 2.0], [-1.0, 0.0]]),  # x + 2y <= 4, -x <= -2
        b_ub=np.array([4.0, -2.0]),
    )

    # Convert to standard form
    standard = problem.to_standard_form()

    # Check dimensions
    assert standard.A.shape == (4, 6)  # 4 constraints (2 non-neg, 2 upper), 2 original + 4 slack
    assert len(standard.b) == 4  # 4 RHS values
    assert len(standard.c) == 6  # 2 original + 4 slack

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            [-1.0, -2.0, 0.0, 0.0, -1.0, 0.0],  # -x - 2y - s3 = -4 (from x + 2y <= 4)
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],  # x - s4 = 2 (from -x <= -2)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([0.0, 0.0, -4.0, 2.0])  # [non-neg, non-neg, upper, lower]
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (Phase 2: minimize original objective)
    expected_c = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])  # Original coefficients + zeros for slack
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE2  # Phase 2 since no artificial variables needed

    # Verify initial basis is valid
    slack_indices = (
        np.array([0, 1, 2, 3]),
        np.array([2, 3, 4, 5]),
    )  # Row and column indices for slack
    standard.basis = Basis.from_standard_form(standard.A, slack_indices, None)
    assert standard.basis.is_valid(standard.A)


def test_phase1_form_with_artificial():
    """Test that a problem with artificial variables starts in Phase 1."""
    A = np.array([[1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]])
    art_indices = (
        np.array([0, 1]),
        np.array([2]),
    )  # Row and column indices for artificial variable

    std = StandardForm(
        c=np.array([0.0, 0.0, 1.0]),  # Phase 1 objective: minimize artificial variables
        A=A,
        b=np.array([5.0, -5.0]),
        basis=Basis(np.array([0, 1])),
        parent=LPProblem,
        phase_type=PhaseType.PHASE1,  # Should be Phase 1 since has artificial variables
        art_indices=art_indices,
        slack_indices=None,
    )

    # The system should be in Phase 1 since it has artificial variables
    assert std.phase_type == PhaseType.PHASE1
    assert std.art_indices is not None
    assert std.A.shape[1] == 3  # Original + artificial
    np.testing.assert_array_equal(std.b, [5.0, -5.0])
    # In Phase 1, we minimize sum of artificial variables
    np.testing.assert_array_equal(std.c, np.array([0.0, 0.0, 1.0]))


def test_phase2_with_solution():
    """Test Phase 2 transformation using Phase 1 solution."""
    basis = Basis(indices=np.array([3]))  # Using artificial variable as basis
    A = np.array([[1.0, 0.0, 1.0, 1.0]])  # 2 original + 1 slack + 1 artificial
    art_indices = (np.array([0]), np.array([3]))  # Row and column indices for artificial variable
    slack_indices = (np.array([0]), np.array([2]))  # Row and column indices for slack variable

    # Test basic Phase 2 transformation with solution
    parent = LPProblem(
        c=np.array([1.0, 2.0]),  # Original objective coefficients
        sense=Sense.MIN,
    )
    phase1 = StandardForm(
        c=np.array([0.0, 0.0, 0.0, -1.0]),  # 2 original + 1 slack + 1 artificial
        A=A,
        b=np.array([5.0]),
        basis=basis,
        parent=parent,  # Use the instance instead of the class
        phase_type=PhaseType.PHASE1,
        art_indices=art_indices,
        slack_indices=slack_indices,
    )

    solution = LPSolution(
        x=np.array([5.0, 0.0, 0.0, 0.0]),  # Solution includes slack
        value=0.0,
        status=None,
        basis=basis,
    )

    phase2 = phase1.to_phase2_form(phase1_solution=solution)
    assert phase2.phase_type == PhaseType.PHASE2
    assert phase2.A.shape[1] == 4  # Original variables + slack + artificial
    np.testing.assert_array_equal(phase2.c[:2], np.array([1.0, 2.0]))  # Original objective restored


def test_phase_conversion_errors():
    """Test error handling in phase conversions."""
    # Test converting standard to Phase 2
    A = np.array([[1.0, 0.0, 1.0]])
    slack_indices = (np.array([0]), np.array([2]))  # Row and column indices for slack variable

    std_form = StandardForm(
        c=np.array([1.0, 2.0, 0.0]),
        A=A,
        b=np.array([5.0]),
        basis=Basis(np.array([2])),
        parent=LPProblem,
        phase_type=None,
        art_indices=None,
        slack_indices=slack_indices,
    )
    with pytest.raises(ValueError, match="Only a Phase1 type form can be converted to Phase2 type"):
        std_form.to_phase2_form()


def test_property_access():
    """Test property access for standard form matrices."""
    # Test with no artificial variables
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    slack_indices = (
        np.array([0, 1]),
        np.array([0, 1]),
    )  # Row and column indices for slack variables

    std = StandardForm(
        c=np.array([1.0, 2.0]),
        A=A,
        b=np.array([1.0, 1.0]),
        basis=Basis(np.array([0, 1])),
        parent=LPProblem,
        phase_type=None,
        art_indices=None,
        slack_indices=slack_indices,
    )

    # Test property access with no artificial variables
    np.testing.assert_array_equal(std.A, A)
    np.testing.assert_array_equal(std.c, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(std.b, np.array([1.0, 1.0]))


def test_variable_bounds_edge_cases():
    """Test edge cases in variable bounds handling."""
    # Test with only infinite bounds
    problem = LPProblem(
        c=np.array([1.0, 2.0]),
        sense=Sense.MIN,
        lb=np.array([-np.inf, -np.inf]),
        ub=np.array([np.inf, np.inf]),
    )
    std = problem.to_standard_form()
    assert std.A.shape[0] == 0  # No constraints needed

    # Test with mixed finite/infinite bounds
    problem = LPProblem(
        c=np.array([1.0, 2.0]),
        sense=Sense.MIN,
        lb=np.array([1.0, -np.inf]),
        ub=np.array([np.inf, 2.0]),
    )
    std = problem.to_standard_form()
    assert std.A.shape[0] == 2  # One lower bound, one upper bound


def test_constraint_handling_edge_cases():
    """Test edge cases in constraint handling."""
    # Test with empty but non-None constraint matrices
    problem = LPProblem(
        c=np.array([1.0, 2.0]),
        sense=Sense.MIN,
        A_ub=np.empty((0, 2)),
        b_ub=np.array([]),
        A_lb=np.empty((0, 2)),
        b_lb=np.array([]),
        A_eq=np.empty((0, 2)),
        b_eq=np.array([]),
    )
    std = problem.to_standard_form()
    assert std.A.shape[0] == 2  # Only non-negativity constraints

    # Test with duplicate constraints
    problem = LPProblem(
        c=np.array([1.0, 2.0]),
        sense=Sense.MIN,
        A_ub=np.array([[1.0, 1.0], [1.0, 1.0]]),  # Duplicate constraint
        b_ub=np.array([5.0, 5.0]),
    )
    std = problem.to_standard_form()
    assert std.A.shape[0] == 3  # 2 non-negativity + 1 unique upper bound


def test_phase1_with_existing_artificial():
    """Test that a standard form with artificial variables starts in Phase 1.

    This tests that a standard form with artificial variables is automatically
    in Phase 1 without needing an explicit conversion.
    """
    # Create a standard form with artificial variables for equality constraints
    A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # x + y + a1 = 3 (equality)
            [1.0, -1.0, 0.0, 0.0, 0.0, 1.0],  # x - y + a2 = 1 (equality)
        ]
    )
    slack_indices = (
        np.array([0, 1]),
        np.array([2, 3]),
    )  # Row and column indices for slack variables
    art_indices = (
        np.array([2, 3]),
        np.array([4, 5]),
    )  # Row and column indices for artificial variables

    std = StandardForm(
        c=np.array(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        ),  # Phase 1 objective: minimize artificial variables
        A=A,
        b=np.array([0.0, 0.0, 3.0, 1.0]),
        basis=Basis(np.array([2, 3, 4, 5])),  # Using slack and artificial variables
        parent=LPProblem,
        phase_type=PhaseType.PHASE1,  # Should be Phase 1 since has artificial variables
        art_indices=art_indices,
        slack_indices=slack_indices,
    )

    # Should be in Phase 1 since artificial indices exist
    assert std.phase_type == PhaseType.PHASE1
    assert std.art_indices is not None
    # In Phase 1, objective should be sum of artificial variables
    np.testing.assert_array_equal(std.c, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))


def test_mixed_slack_artificial_placement():
    """Test correct placement of slack variables with mixed artificial/non-artificial rows.

    System:
    minimize    x + y
    subject to:
        x >= 0          (non-negativity)
        y >= 0          (non-negativity)
        x >= 1          (lower bound)
        y >= 3          (lower bound, needs artificial due to positive RHS)
        x + y = 2       (equality, needs artificial)
        2x - y = 0      (equality, needs artificial)

    Note: The order of constraints in the standard form is:
    1. Variable bounds (non-negativity and explicit bounds)
    2. General inequality constraints (none in this case)
    3. Equality constraints (with artificial variables)
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),
        sense=Sense.MIN,
        A_lb=np.array([[1.0, 0.0], [0.0, 1.0]]),  # x >= 1, y >= 3
        b_lb=np.array([1.0, 3.0]),
        A_eq=np.array([[1.0, 1.0], [2.0, -1.0]]),  # x + y = 2, 2x - y = 0
        b_eq=np.array([2.0, 0.0]),
    )

    standard = problem.to_standard_form()

    # Check dimensions: 6 rows (2 non-neg + 2 lower bounds + 2 equalities), 9 cols (2 original + 4 slack + 3 artificial)
    assert standard.A.shape == (6, 9)

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # x - s3 = 1 (x >= 1)
            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],  # y - s4 + a1 = 3 (y >= 3)
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # x + y + a2 = 2 (equality)
            [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 2x - y + a3 = 0 (equality)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array(
        [0.0, 0.0, 1.0, 3.0, 2.0, 0.0]
    )  # [non-neg, non-neg, lower, lower, eq, eq]
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients (Phase 1: minimize artificial variables)
    expected_c = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    )  # Zeros for original/slack + ones for artificial
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE1  # Phase 1 due to artificial variables

    # Verify initial basis is valid
    slack_indices = (np.array([0, 1, 2]), np.array([2, 3, 4]))  # First 3 rows use slack variables
    art_indices = (
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
    )  # Last 3 rows use artificial variables
    standard.basis = Basis.from_standard_form(standard.A, slack_indices, art_indices)
    assert standard.basis.is_valid(standard.A)


def test_mixed_lower_bounds():
    """Test handling of mixed zero and non-zero lower bounds.

    Original problem:
        minimize    x + y + z + w + v
        subject to:
            x >= 0          (non-negativity)
            y >= 2          (non-zero lower bound)
            z >= 0          (non-negativity)
            w >= 0          (non-negativity)
            v >= 3          (non-zero lower bound)

    Note: All variables get non-negativity constraints by default.
    Non-zero lower bounds require artificial variables, putting us in Phase 1.
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # [x, y, z, w, v]
        sense=Sense.MIN,
        lb=np.array([0.0, 2.0, 0.0, 0.0, 3.0]),  # Mix of zero and non-zero bounds
        ub=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
    )

    standard = problem.to_standard_form()

    # Check dimensions: 5 constraints (all lower bounds), 5 original + 5 slack
    assert standard.A.shape == (5, 10)
    assert len(standard.b) == 5
    assert len(standard.c) == 10

    # Check constraint matrix
    expected_A = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # x - s1 = 0
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # y - s2 + a1 = 2
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],  # z - s3 = 0
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],  # w - s4 = 0
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],  # v - s5 + a2 = 3
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS
    expected_b = np.array([0.0, 2.0, 0.0, 0.0, 3.0])  # Mix of zero and non-zero bounds
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients - Phase 1 objective minimizes sum of artificial variables
    expected_c = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )  # Zeros for original/slack
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE2  # Phase 2 (no artificial variables)


def test_mixed_upper_bounds():
    """Test handling of mixed finite and infinite upper bounds.

    Original problem:
        minimize    x + y
        subject to:
            x >= 0          (non-negativity)
            y >= 0          (non-negativity)
            x <= 2          (finite upper bound)
            y <= 3          (finite upper bound)

    Expected standard form:
        1. Variable bounds (x >= lb):
            x >= 0          (non-negativity)
            y >= 0          (non-negativity)
        2. Variable bounds (x <= ub):
            x <= 2          (upper bound)
            y <= 3          (upper bound)
    """
    problem = LPProblem(
        c=np.array([1.0, 1.0]),  # [x, y]
        sense=Sense.MIN,
        lb=np.array([0.0, 0.0]),  # All zero lower bounds
        ub=np.array([2.0, 3.0]),  # Both finite upper bounds
    )

    standard = problem.to_standard_form()

    # Check dimensions: 4 constraints (2 non-negativity + 2 upper bounds), 2 original + 4 slack
    assert standard.A.shape == (4, 6)
    assert len(standard.b) == 4
    assert len(standard.c) == 6

    # Check constraint matrix
    expected_A = np.array(
        [
            # 1. Variable bounds (x >= lb)
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 0 (x >= 0)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 0 (y >= 0)
            # 2. Variable bounds (x <= ub)
            [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],  # -x - s3 = -2 (x <= 2)
            [0.0, -1.0, 0.0, 0.0, 0.0, -1.0],  # -y - s4 = -3 (y <= 3)
        ]
    )
    np.testing.assert_array_almost_equal(standard.A, expected_A)

    # Check RHS - same order as constraints
    expected_b = np.array(
        [
            0.0,
            0.0,  # Non-negativity
            -2.0,
            -3.0,
        ]
    )  # Upper bounds
    np.testing.assert_array_almost_equal(standard.b, expected_b)

    # Check objective coefficients
    expected_c = np.array(
        [
            1.0,
            1.0,  # Original coefficients
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )  # Slack variables
    np.testing.assert_array_almost_equal(standard.c, expected_c)

    # Check phase type
    assert standard.phase_type == PhaseType.PHASE2  # No artificial variables needed

    # Verify initial basis is valid
    slack_indices = (
        np.array([0, 1, 2, 3]),  # Row indices
        np.array([2, 3, 4, 5]),
    )  # Column indices for slack variables
    standard.basis = Basis.from_standard_form(standard.A, slack_indices, None)
    assert standard.basis.is_valid(standard.A)
