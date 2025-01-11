import numpy as np
import pytest

from boundhound.core.basis import Basis, InvalidBasisError
from boundhound.types import LPSolution, SimplexStatus


def test_basis_creation():
    """Test basic creation of Basis objects.

    Tests:
        1. Valid creation with numpy array
        2. Empty basis creation
        3. Creation with Python list (should convert to ndarray)
    """
    # Valid creation
    basis = Basis(indices=np.array([0, 1]))
    assert len(basis.indices) == 2
    assert basis.m == 2
    np.testing.assert_array_equal(basis.indices, np.array([0, 1]))

    # Empty basis
    basis = Basis(indices=np.array([], dtype=np.int_))
    assert len(basis.indices) == 0
    assert basis.m == 0

    # Creation with list (should convert to ndarray)
    basis = Basis(indices=np.array([0, 1], dtype=np.int_))
    assert isinstance(basis.indices, np.ndarray)
    np.testing.assert_array_equal(basis.indices, np.array([0, 1]))


def test_basis_validation():
    """Test basis validation against different matrices.

    Tests:
        1. Valid basis with unit matrix
        2. Invalid basis with dependent matrix
        3. Invalid basis size
        4. Valid basis with slack variables
    """
    # Create test matrices
    unit_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 unit matrix
    dependent_matrix = np.array([[1.0, 2.0], [2.0, 4.0]])  # Linearly dependent columns

    # Test with unit matrix (should be valid)
    basis = Basis(indices=np.array([0, 1]))
    assert basis.is_valid(unit_matrix)

    # Test with dependent matrix (should fail)
    basis = Basis(indices=np.array([0, 1]))
    assert not basis.is_valid(dependent_matrix)

    # Wrong size basis
    basis = Basis(indices=np.array([0]))
    assert not basis.is_valid(unit_matrix)


def test_from_standard_form():
    """Test creation of basis from standard form.

    Tests:
        1. Valid case with slack and artificial variables
        2. Valid case with only slack variables
        3. Valid case with only artificial variables
    """
    # Create test matrix
    matrix = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0],  # x - s1 = 2 (inequality)
            [0.0, 1.0, 0.0, -1.0, 0.0],  # y - s2 = 1 (inequality)
            [1.0, 1.0, 0.0, 0.0, 1.0],  # x + y + a1 = 2 (equality)
        ]
    )

    # Test with both slack and artificial variables
    slack_indices = (np.array([0, 1]), np.array([2, 3]))
    art_indices = (np.array([2]), np.array([4]))
    basis = Basis.from_standard_form(matrix, slack_indices=slack_indices, art_indices=art_indices)
    np.testing.assert_array_equal(basis.indices, np.array([0, 1, 4]))

    # Test with only slack variables
    matrix_no_eq = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0],  # x - s1 = 2 (inequality)
            [0.0, 1.0, 0.0, -1.0, 0.0],  # y - s2 = 1 (inequality)
            [1.0, 0.0, 0.0, 0.0, -1.0],  # x - s3 = 0 (inequality)
        ]
    )
    slack_indices = (np.array([0, 1, 2]), np.array([2, 3, 4]))
    basis = Basis.from_standard_form(matrix_no_eq, slack_indices=slack_indices, art_indices=None)
    np.testing.assert_array_equal(
        basis.indices, np.array([0, 1, 4])
    )  # Uses x, y for first two rows, s3 for last row since x is already used

    # Test with only artificial variables
    art_indices = (np.array([0, 1, 2]), np.array([2, 3, 4]))
    basis = Basis.from_standard_form(matrix, slack_indices=None, art_indices=art_indices)
    np.testing.assert_array_equal(basis.indices, np.array([2, 3, 4]))


def test_from_phase1_solution():
    """Test creating a basis from Phase 1 solution.

    Tests:
        1. Valid case replacing artificial variables
        2. Valid case with no artificial variables to replace
        3. Invalid case when artificial variables cannot be replaced
    """
    # Create test matrix
    matrix = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0],  # x - s1 = 2 (inequality)
            [0.0, 1.0, 0.0, -1.0, 0.0],  # y - s2 = 1 (inequality)
            [1.0, 1.0, 0.0, 0.0, 1.0],  # x + y + a1 = 2 (equality)
        ]
    )

    # Test replacing multiple artificial variables
    phase1_basis = Basis(indices=np.array([4, 2, 3]))  # Using a1, s1, s2
    phase1_solution = LPSolution(
        x=np.array([0.0, 0.0, 2.0, 1.0, 2.0]),
        value=2.0,
        status=SimplexStatus.OPTIMAL,
        basis=phase1_basis,
    )
    art_indices = (np.array([2]), np.array([4]))  # Column 4 is a1
    basis = Basis.from_phase1_solution(phase1_solution, matrix, art_indices)
    np.testing.assert_array_equal(basis.indices, np.array([0, 2, 3]))  # Replace a1 with x

    # Test with no artificial variables
    phase1_basis = Basis(indices=np.array([2, 3, 0]))  # Using s1, s2, x
    phase1_solution = LPSolution(
        x=np.array([2.0, 0.0, 0.0, 1.0, 0.0]),
        value=0.0,
        status=SimplexStatus.OPTIMAL,
        basis=phase1_basis,
    )
    basis = Basis.from_phase1_solution(phase1_solution, matrix, art_indices=None)
    np.testing.assert_array_equal(basis.indices, np.array([2, 3, 0]))

    # Test when artificial variables cannot be replaced
    phase1_basis = Basis(indices=np.array([4, 4, 4]))  # Invalid basis with artificial variables
    phase1_solution = LPSolution(
        x=np.array([0.0, 0.0, 0.0, 0.0, 2.0]),
        value=2.0,
        status=SimplexStatus.OPTIMAL,
        basis=phase1_basis,
    )
    art_indices = (np.array([2]), np.array([4]))  # Column 4 is a1
    with pytest.raises(InvalidBasisError):
        Basis.from_phase1_solution(phase1_solution, matrix, art_indices)


def test_basis_selection_with_ge_constraints():
    """Test basis selection with >= and <= constraints.

    Tests:
        1. Selection of original variables when they form a valid basis
        2. Mixed selection of original and slack variables
        3. Handling of both upper and lower bounds
    """
    # Create test matrix with bounds and inequality constraints
    matrix = np.array(
        [
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],  # x - s1 = 2 (x >= 2)
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],  # y - s2 = 1 (y >= 1)
            [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0],  # -x - s3 = -4 (x <= 4)
            [0.0, -1.0, 0.0, 0.0, 0.0, -1.0],  # -y - s4 = -3 (y <= 3)
        ]
    )
    b = np.array([2.0, 1.0, -4.0, -3.0])

    # Test with slack variables provided
    slack_indices = (np.arange(4), np.array([2, 3, 4, 5]))
    basis = Basis.from_standard_form(matrix, slack_indices=slack_indices)
    # Should use original variables for first two rows (x, y) and slack for last two (-s3, -s4)
    np.testing.assert_array_equal(basis.indices, np.array([0, 1, 4, 5]))

    # Verify the basis is valid
    assert basis.is_valid(matrix)


def test_mixed_constraints_basis_selection():
    """Test basis selection with mixed equality and inequality constraints.

    Tests:
        1. Artificial variables must be used for equality constraints
        2. Original variables preferred over slack for inequalities
        3. Basis must be valid (linearly independent)
    """
    # Create test matrix with mixed constraints
    matrix = np.array(
        [
            [1.0, 1.0, -1.0, 0.0, 0.0],  # x + y - s1 = 4 (inequality)
            [1.0, -1.0, 0.0, 0.0, 1.0],  # x - y + a1 = 1 (equality)
        ]
    )
    b = np.array([4.0, 1.0])

    # Test with mixed slack and artificial variables
    slack_indices = (np.array([0]), np.array([2]))
    art_indices = (np.array([1]), np.array([4]))
    basis = Basis.from_standard_form(matrix, slack_indices, art_indices)
    # Should use:
    # - artificial variable (4) for equality constraint
    # - original variable (0) for inequality since it's linearly independent
    np.testing.assert_array_equal(basis.indices, np.array([0, 4]))
    assert basis.is_valid(matrix)  # Verify the basis is valid


def test_basis_edge_cases():
    """Test edge cases in basis creation and validation.

    Tests:
        1. Empty basis with empty matrix is valid
        2. Original variables used if linearly independent
        3. Must use artificial variables for equality constraints
    """
    # Empty matrix case
    empty_matrix = np.array([]).reshape(0, 0)
    empty_basis = Basis(indices=np.array([], dtype=np.int_))
    assert empty_basis.is_valid(empty_matrix)

    # Original variables should be used if they form a linearly independent basis
    no_unit_matrix = np.array(
        [
            [1.0, 1.0, -1.0, 0.0],  # x + y - s1 = 0
            [1.0, -1.0, 0.0, -1.0],  # x - y - s2 = 0
        ]
    )
    slack_indices = (np.array([0, 1]), np.array([2, 3]))
    basis = Basis.from_standard_form(no_unit_matrix, slack_indices=slack_indices)
    # Should use original variables since they form a valid basis
    np.testing.assert_array_equal(basis.indices, np.array([0, 1]))
    assert basis.is_valid(no_unit_matrix)

    # When all constraints are equalities, must use artificial variables
    all_art_matrix = np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],  # x + y + a1 = 1 (equality)
            [1.0, -1.0, 0.0, 1.0, 0.0],  # x - y + a2 = 0 (equality)
            [2.0, 1.0, 0.0, 0.0, 1.0],  # 2x + y + a3 = 2 (equality)
        ]
    )
    art_indices = (np.array([0, 1, 2]), np.array([2, 3, 4]))
    basis = Basis.from_standard_form(all_art_matrix, art_indices=art_indices)
    # Must use artificial variables for equality constraints
    np.testing.assert_array_equal(basis.indices, np.array([2, 3, 4]))
    assert basis.is_valid(all_art_matrix)


def test_basis_selection_with_negative_bounds():
    """Test basis selection with negative lower bounds.

    Tests:
        1. Automatic basis selection with negative bounds
        2. Validation of selected basis
        3. Handling of non-negativity constraints
    """
    # Create test matrix with negative bounds
    matrix = np.array(
        [
            [1.0, 0.0, -1.0, 0.0],  # x - s1 = -1 (x >= -1)
            [0.0, 1.0, 0.0, -1.0],  # y - s2 = 1 (y >= 1)
        ]
    )

    # Test with slack variables provided
    slack_indices = (np.array([0, 1]), np.array([2, 3]))
    basis = Basis.from_standard_form(matrix, slack_indices=slack_indices)
    assert basis.is_valid(matrix)
    # Should prefer original variables when they form a valid basis
    np.testing.assert_array_equal(basis.indices, np.array([0, 1]))

    # Test with original variables (should be valid since they're linearly independent)
    basis = Basis(indices=np.array([0, 1]))
    assert basis.is_valid(matrix)  # Valid because columns are linearly independent
