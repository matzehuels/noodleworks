from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Set, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from boundhound.core.basis import Basis
from boundhound.types import LPSolution, PhaseType, Sense

T = TypeVar("T", bound=np.floating)

logger = logging.getLogger(__name__)


class LPProblem(object):
    parent = None

    def __init__(
        self,
        c: NDArray[T],
        sense: Sense = Sense.MIN,
        A_ub: Optional[NDArray[T]] = None,
        b_ub: Optional[NDArray[T]] = None,
        A_lb: Optional[NDArray[T]] = None,
        b_lb: Optional[NDArray[T]] = None,
        A_eq: Optional[NDArray[T]] = None,
        b_eq: Optional[NDArray[T]] = None,
        lb: Optional[NDArray[T]] = None,
        ub: Optional[NDArray[T]] = None,
    ) -> None:
        """
        Note: Non-negativity constraints (x >= 0) are added by default for all variables
        unless overridden by explicit bounds.
        """
        self.sense = sense
        self._c = c
        self.A_ub = A_ub if A_ub is not None else np.zeros((0, self.n))
        self.b_ub = b_ub if b_ub is not None else np.array([])
        self.A_lb = A_lb if A_lb is not None else np.zeros((0, self.n))
        self.b_lb = b_lb if b_lb is not None else np.array([])
        self.A_eq = A_eq if A_eq is not None else np.zeros((0, self.n))
        self.b_eq = b_eq if b_eq is not None else np.zeros((0, self.n))
        self.lb = lb if lb is not None else np.full(self.n, 0.0)  # Default non-negativity
        self.ub = ub if ub is not None else np.full(self.n, np.inf)  # Default to no upper bounds

    def __repr__(self) -> str:
        """Return a string that could be used to recreate the object."""
        args = [f"c={self._c!r}", f"sense={self.sense!r}"]

        if self.A_ub.size > 0:
            args.append(f"A_ub={self.A_ub!r}")
            args.append(f"b_ub={self.b_ub!r}")
        if self.A_lb.size > 0:
            args.append(f"A_lb={self.A_lb!r}")
            args.append(f"b_lb={self.b_lb!r}")
        if self.A_eq.size > 0:
            args.append(f"A_eq={self.A_eq!r}")
            args.append(f"b_eq={self.b_eq!r}")
        if not np.all(self.lb == 0):
            args.append(f"lb={self.lb!r}")
        if not np.all(np.isinf(self.ub)):
            args.append(f"ub={self.ub!r}")

        return f"InequalityForm({', '.join(args)})"

    @property
    def obj_factor(self) -> float:
        return -1 if self.sense == Sense.MAX else 1

    @property
    def c(self) -> NDArray[T]:
        return self.obj_factor * self._c

    @property
    def n(self) -> int:
        """Number of variables in the problem."""
        return len(self.c)

    def _remove_redundant_constraints(
        self, A: NDArray[T], b: NDArray[T]
    ) -> Tuple[NDArray[T], NDArray[T]]:
        """Remove redundant constraints by normalizing and finding unique rows."""
        if A.size == 0:
            return A, b

        # Normalize rows to handle redundant constraints
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_rows = A / norms
        normalized_b = b / norms.flatten()

        # Get unique rows while preserving order
        _, unique_idx = np.unique(
            np.column_stack([normalized_rows, normalized_b]), axis=0, return_index=True
        )
        unique_idx.sort()  # Sort to maintain original order

        return A[unique_idx], b[unique_idx]

    def to_standard_form(self) -> StandardForm:
        """Convert an inequality form linear programming problem to standard form.

        The standard form has the following properties:
        1. Variable bounds (x >= lb) in variable order
        2. Variable bounds (x <= ub) in variable order
        3. General lower bounds (Ax >= b)
        4. General upper bounds (Ax <= b)
        5. Equality constraints (Ax = b)

        Variables in standard form:
        1. Original variables (x1, x2, ...)
        2. Slack variables for inequalities
        3. Artificial variables for equalities and infeasible >= constraints

        All slack variables are negative since all constraints are converted to >= form:
        - Lower bounds (x >= b) become x - s = b
        - Upper bounds (x <= b) become -x - s = -b (after converting to -x >= -b)
        - Equalities (x = b) get artificial variables
        - Lower bounds (Ax >= b) with b > 0 get artificial variables if no trivial solution exists
        """
        # Handle variable bounds (get slack variables)
        bound_constraints = []
        bound_rhs = []
        for mask, matrix, values, negate in [
            (np.isfinite(self.lb), np.eye(self.n), self.lb, False),  # x >= lb
            (np.isfinite(self.ub), np.eye(self.n), self.ub, True),  # x <= ub
        ]:
            bound_constraints.append(-matrix[mask] if negate else matrix[mask])
            bound_rhs.extend(-values[mask] if negate else values[mask])

        # Handle inequality constraints (get slack variables)
        ineq_constraints = []
        ineq_rhs = []
        needs_artificial = []  # Track which rows need artificial variables

        # Handle lower bounds (Ax >= b)
        if self.A_lb.size > 0:
            A_lb, b_lb = self._remove_redundant_constraints(self.A_lb, self.b_lb)
            positive_rhs_mask = b_lb > 0  # Positive RHS can be infeasible at origin!
            if np.any(positive_rhs_mask):
                positive_coef_sums = np.sum(np.where(A_lb > 0, A_lb, 0), axis=1)
                infeasible_mask = (positive_coef_sums < b_lb) & positive_rhs_mask
                infeasible_indices = np.where(infeasible_mask)[0]
                needs_artificial.extend(len(ineq_constraints) + infeasible_indices)
            ineq_constraints.append(A_lb)
            ineq_rhs.extend(b_lb)

        # Handle upper bounds (Ax <= b)
        if self.A_ub.size > 0:
            A_ub, b_ub = self._remove_redundant_constraints(self.A_ub, self.b_ub)
            ineq_constraints.append(-A_ub)  # Convert to >= form
            ineq_rhs.extend(-b_ub)

        # Handle equality constraints (get artificial variables)
        eq_constraints = []
        eq_rhs = []
        if self.A_eq.size > 0:
            A_eq, b_eq = self._remove_redundant_constraints(self.A_eq, self.b_eq)
            eq_constraints.extend([A_eq])
            eq_rhs.extend(b_eq.tolist())

        # Stack all constraints in order: inequalities first, then equalities
        A = np.vstack([*bound_constraints, *ineq_constraints, *eq_constraints])
        b = np.array([*bound_rhs, *ineq_rhs, *eq_rhs])

        # Calculate dimensions
        n_bound = len(bound_rhs)  # Number of bound constraints
        n_ineq = len(ineq_rhs)  # Number of inequality constraints
        n_eq = len(eq_rhs)  # Number of equality constraints
        n_art_ineq = len(needs_artificial)  # Constraints needing artificial variables
        n_slack = n_bound + n_ineq  # Total slack variables
        n_art = n_eq + n_art_ineq  # Total artificial variables
        m = len(b)  # Total constraints

        # Create extended matrices
        total_cols = self.n + n_slack + n_art
        A_ext = np.zeros((m, total_cols))
        c_ext = np.zeros(total_cols)
        b_ext = b.copy()

        # Copy original constraints
        A_ext[:, : self.n] = A

        # Add slack variables for inequalities
        if n_slack > 0:
            slack_rows = np.arange(n_bound + n_ineq)
            slack_cols = np.arange(self.n, self.n + n_slack)
            A_ext[slack_rows, slack_cols] = -1
            slack_indices = (slack_rows, slack_cols)
        else:
            slack_indices = None

        # Add artificial variables for equalities and infeasible inequalities
        art_rows = []
        art_cols = []

        # First add artificial variables for infeasible inequalities
        if n_art_ineq > 0:
            ineq_art_rows = np.array([n_bound + i for i in needs_artificial])
            ineq_art_cols = np.arange(self.n + n_slack, self.n + n_slack + n_art_ineq)
            A_ext[ineq_art_rows, ineq_art_cols] = 1
            art_rows.extend(ineq_art_rows)
            art_cols.extend(ineq_art_cols)

        # Then add artificial variables for equalities
        if n_eq > 0:
            eq_art_rows = np.arange(n_bound + n_ineq, m)
            eq_art_cols = np.arange(self.n + n_slack + n_art_ineq, total_cols)
            A_ext[eq_art_rows, eq_art_cols] = 1
            art_rows.extend(eq_art_rows)
            art_cols.extend(eq_art_cols)

        art_indices = (np.array(art_rows), np.array(art_cols)) if art_rows else None

        # Set objective coefficients
        if n_art > 0:  # Phase 1
            c_ext[self.n + n_slack :] = 1  # Minimize sum of artificial variables
            phase_type = PhaseType.PHASE1
        else:  # Phase 2
            c_ext[: self.n] = self.c
            phase_type = PhaseType.PHASE2

        return StandardForm(
            c=c_ext,
            A=A_ext,
            b=b_ext,
            basis=Basis.from_standard_form(A_ext, slack_indices, art_indices),
            parent=self,
            phase_type=phase_type,
            art_indices=art_indices,
            slack_indices=slack_indices,
        )

    def copy(self) -> LPProblem:
        """Create a deep copy of this InequalityForm.

        Returns:
            A new InequalityForm instance with copies of all arrays.
        """
        return LPProblem(
            c=self._c.copy(),
            sense=self.sense,
            A_ub=self.A_ub.copy() if self.A_ub is not None else None,
            b_ub=self.b_ub.copy() if self.b_ub is not None else None,
            A_lb=self.A_lb.copy() if self.A_lb is not None else None,
            b_lb=self.b_lb.copy() if self.b_lb is not None else None,
            A_eq=self.A_eq.copy() if self.A_eq is not None else None,
            b_eq=self.b_eq.copy() if self.b_eq is not None else None,
            lb=self.lb.copy() if self.lb is not None else None,
            ub=self.ub.copy() if self.ub is not None else None,
        )


class StandardForm(object):
    """A linear program in standard form, ready for simplex method solution."""

    def __init__(
        self,
        c: NDArray[T],
        A: NDArray[T],
        b: NDArray[T],
        basis: Basis,
        phase_type: Optional[PhaseType],
        parent: StandardForm | LPProblem,
        art_indices: Optional[Tuple[NDArray[np.int_], NDArray[np.int_]]],
        slack_indices: Optional[Tuple[NDArray[np.int_], NDArray[np.int_]]],
    ) -> None:
        self.c = c
        self.A = A
        self.b = b
        self.basis = basis
        self.phase_type = phase_type
        self.parent = parent
        self.art_indices = art_indices
        self.slack_indices = slack_indices

    def __repr__(self) -> str:
        """Return a string that could be used to recreate the object."""
        args = [
            f"c={self.c!r}",
            f"A={self.A!r}",
            f"b={self.b!r}",
            f"basis={self.basis!r}",
            f"phase_type={self.phase_type!r}",
            f"parent={self.parent!r}",
            f"art_indices={self.art_indices!r}",
            f"slack_indices={self.slack_indices!r}",
        ]
        return f"StandardForm({', '.join(args)})"

    def to_phase2_form(self, phase1_solution: Optional[LPSolution] = None) -> StandardForm:
        """Convert Phase 1 problem to Phase 2 (original optimization problem).

        Strategy:
        1. Keep the same A matrix and b vector
        2. Restore original objective coefficients from parent
        3. Keep artificial variables in matrix but with large positive coefficients
           to prevent them from re-entering the basis
        """
        if self.phase_type == PhaseType.PHASE2:
            return self
        elif self.phase_type is None:
            raise ValueError("Only a Phase1 type form can be converted to Phase2 type")

        # Restore original objective coefficients, extended with large positive values for artificial variables
        c_phase2 = np.zeros_like(self.c)
        c_phase2[: len(self.parent.c)] = self.parent.c
        if self.art_indices is not None:
            c_phase2[self.art_indices[1]] = 1e6  # Large positive value to prevent re-entry to basis

        return StandardForm(
            c=c_phase2,
            A=self.A,  # Keep the same A matrix
            b=self.b,
            basis=Basis.from_phase1_solution(phase1_solution, self.A, self.art_indices),
            parent=self,
            phase_type=PhaseType.PHASE2,
            art_indices=None,  # Still mark as no artificial variables for Phase 2
            slack_indices=self.slack_indices,
        )


@dataclass
class MILPProblem:
    """A mixed-integer linear programming problem."""

    lp: LPProblem
    integer_vars: Set[int]
