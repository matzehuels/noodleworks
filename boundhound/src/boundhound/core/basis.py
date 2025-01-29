from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from boundhound.types import LPSolution

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=np.floating)


class InvalidBasisError(Exception):
    """Raised when a basis is invalid.

    This error occurs when:
    1. The basis matrix is not full rank
    2. The number of basis vectors doesn't match the number of constraints
    3. No valid basis can be found during construction
    """

    pass


class Basis:
    """A basis for a linear program, represented by column indices of the constraint matrix A.

    In the revised simplex method, a basis identifies a vertex of the feasible polytope
    by selecting m linearly independent columns from the constraint matrix A.
    """

    def __init__(self, indices: NDArray[np.int_] | Sequence[int]) -> None:
        """Initialize a basis with the given column indices."""
        self.indices: NDArray[np.int_] = (
            indices if isinstance(indices, np.ndarray) else np.asarray(indices, dtype=np.int_)
        )

    def __repr__(self) -> str:
        """Return a string representation of the basis."""
        return f"Basis(indices={self.indices!r})"

    def copy(self) -> Basis:
        """Create a deep copy of the basis."""
        return Basis(indices=self.indices.copy())

    @property
    def m(self) -> int:
        """Number of constraints (size of basis)."""
        return len(self.indices)

    def is_valid(self, A: NDArray[T]) -> bool:
        """Check if basis matrix is valid according to the revised simplex method.

        A basis is valid if:
        1. The number of basis vectors equals the number of constraints
        2. The basis vectors are linearly independent (full rank)
        """
        m, _ = A.shape
        if self.m != m:
            return False

        # Empty matrix is valid if it has no indices
        if m == 0:
            return True

        # Get basis matrix and check rank
        B = A[:, self.indices]
        return bool(np.linalg.matrix_rank(B) == m)

    @classmethod
    def from_standard_form(
        cls,
        A: NDArray[T],
        slack_indices: tuple[NDArray[np.int_], NDArray[np.int_]] | None = None,
        art_indices: tuple[NDArray[np.int_], NDArray[np.int_]] | None = None,
    ) -> Basis:
        """Create a basis from a standard form problem.

        The method follows a priority order to construct a valid basis:
        1. Use artificial variables (these must be in the basis)
        2. Use original variables with non-zero coefficients
        3. Use slack variables for remaining rows
        """
        m = A.shape[0]
        n_original = A.shape[1] - m if A.shape[1] > m else A.shape[1]
        basis_indices = np.zeros(m, dtype=np.int_)
        used_rows: set[int] = set()

        # Priority 1: First use artificial variables (these must be in the basis)
        if art_indices is not None:
            art_rows, art_cols = art_indices
            basis_indices[art_rows] = art_cols
            used_rows.update(art_rows)

        # Priority 2: Try original variables with non-zero coefficients
        remaining_rows = [i for i in range(m) if i not in used_rows]
        for row in remaining_rows:
            for j in range(n_original):
                if A[row, j] != 0.0:
                    test_indices = basis_indices.copy()
                    test_indices[row] = j
                    # Check if valid for all assigned rows so far
                    assigned_rows = [*list(used_rows), row]
                    test_basis = cls(indices=test_indices[assigned_rows])
                    if test_basis.is_valid(A[assigned_rows, :]):
                        basis_indices[row] = j
                        used_rows.add(row)
                        break

        # Priority 3: Use slack variables for remaining rows
        if slack_indices is not None:
            slack_rows, slack_cols = slack_indices
            remaining_rows = [i for i in range(m) if i not in used_rows]
            # Only use slack variables for unused rows
            remaining_slack_mask = np.isin(slack_rows, remaining_rows)
            remaining_slack_rows = slack_rows[remaining_slack_mask]
            remaining_slack_cols = slack_cols[remaining_slack_mask]
            basis_indices[remaining_slack_rows] = remaining_slack_cols
            used_rows.update(remaining_slack_rows)

        # Verify final basis
        basis = cls(indices=basis_indices)
        if not basis.is_valid(A):
            raise InvalidBasisError("Could not construct a valid basis")

        return basis

    @classmethod
    def from_phase1_solution(
        cls,
        phase1_solution: LPSolution,
        A: NDArray[T],
        art_indices: tuple[NDArray[np.int_], NDArray[np.int_]] | None = None,
    ) -> Basis:
        """Create a Phase 2 basis from a Phase 1 solution.

        This method attempts to replace artificial variables in the basis with
        other available variables while maintaining basis validity.
        """
        basis_indices = phase1_solution.basis.indices.copy()

        # Replace artificial variables if present
        if art_indices is not None:
            art_cols = art_indices[1]
            art_positions = np.nonzero(np.isin(basis_indices, art_cols))[0]

            if len(art_positions) > 0:
                # Try replacing artificial variables with non-artificial ones
                non_art_vars = np.arange(A.shape[1])[~np.isin(np.arange(A.shape[1]), art_cols)]
                for pos in art_positions:
                    available_vars = non_art_vars[~np.isin(non_art_vars, basis_indices)]
                    for var in available_vars:
                        test_indices = basis_indices.copy()
                        test_indices[pos] = var
                        test_basis = cls(indices=test_indices)
                        if test_basis.is_valid(A):
                            basis_indices = test_indices
                            break

        # Create and verify final basis
        basis = cls(indices=basis_indices)
        if not basis.is_valid(A):
            raise InvalidBasisError("Could not construct a valid Phase 2 basis")

        return basis
