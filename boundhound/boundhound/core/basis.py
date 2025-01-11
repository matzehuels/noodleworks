from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from boundhound.types import LPSolution

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=np.floating)


class InvalidBasisError(Exception):
    """Raised when a basis is invalid."""

    pass


class Basis:
    """A basis for a linear program, represented by column indices of the constraint matrix A.

    In the revised simplex method, a basis identifies a vertex of the feasible polytope
    by selecting m linearly independent columns from the constraint matrix A.
    """

    def __init__(self, indices: Union[NDArray[np.int_], Sequence[int]]):
        """Initialize a basis with the given column indices."""
        # Convert indices to numpy array if needed
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices, dtype=np.int_)
        self.indices = indices

    def __repr__(self) -> str:
        """Return a string representation of the basis."""
        return f"Basis(indices={self.indices})"

    def copy(self) -> Basis:
        """Create a copy of the basis."""
        return Basis(indices=self.indices.copy())

    @property
    def m(self) -> int:
        """Number of constraints (size of basis)."""
        return len(self.indices)

    def is_valid(self, A: NDArray[T]) -> bool:
        """Check if basis matrix is valid according to the revised simplex method."""
        m, _ = A.shape
        if self.m != m:
            return False

        # Empty matrix is valid if it has no indices
        if m == 0:
            return True

        # Get basis matrix and check rank
        B = A[:, self.indices]
        if np.linalg.matrix_rank(B) != m:
            return False

        return True

    @classmethod
    def from_standard_form(
        cls,
        A: NDArray[T],
        slack_indices: Optional[Tuple[NDArray[np.int_], NDArray[np.int_]]] = None,
        art_indices: Optional[Tuple[NDArray[np.int_], NDArray[np.int_]]] = None,
    ) -> Basis:
        """Create a basis from a standard form problem.

        Strategy:
        1. Use artificial variables
        2. Use original variables
        3. Use slack variables for remaining rows
        """
        m = A.shape[0]
        n_original = A.shape[1] - m if A.shape[1] > m else A.shape[1]
        basis_indices = np.zeros(m, dtype=np.int_)
        used_rows = set()

        # Priority 1: First use artificial variables (these must be in the basis)
        if art_indices is not None:
            art_rows, art_cols = art_indices
            basis_indices[art_rows] = art_cols
            used_rows.update(art_rows)

        # Priority 2: Try original variables with non-zero coefficients for remaining rows
        remaining_rows = [i for i in range(m) if i not in used_rows]
        for row in remaining_rows:
            for j in range(n_original):
                if A[row, j] != 0.0:
                    test_indices = basis_indices.copy()
                    test_indices[row] = j
                    # Check if valid for all assigned rows so far
                    assigned_rows = list(used_rows) + [row]
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
            raise InvalidBasisError("Could not find valid basis")

        return basis

    @classmethod
    def from_phase1_solution(
        cls,
        phase1_solution: LPSolution,
        A: NDArray[T],
        art_indices: Optional[Tuple[NDArray[np.int_], NDArray[np.int_]]] = None,
    ) -> Basis:
        """Create a Phase 2 basis from a Phase 1 solution."""
        basis_indices = phase1_solution.basis.indices.copy()

        # If no artificial variables or basis is valid, use it directly
        if art_indices is not None:
            # Replace artificial variables with other available variables
            art_cols = art_indices[1]
            art_positions = np.nonzero(np.isin(basis_indices, art_cols))[0]

            if len(art_positions) > 0:
                # Try replacing artificial variables with available non-artificial variables
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

        basis = cls(indices=basis_indices)
        if basis.is_valid(A):
            return basis
        else:
            raise InvalidBasisError("No valid basis found")
