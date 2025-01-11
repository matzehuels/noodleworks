"""Implementation of the two-phase revised simplex method for linear programming.

The simplex method solves linear programs in standard form:
    min c^T x
    s.t. Ax = b
         x >= 0

The two-phase method handles problems requiring artificial variables:
Phase 1: Find initial feasible solution by minimizing sum of artificial variables
Phase 2: Solve original problem using feasible basis from Phase 1

The standard form transformation maintains a specific order:
1. Variable bounds (x >= lb) in variable order
2. Variable bounds (x <= ub) in variable order
3. General lower bounds (Ax >= b)
4. General upper bounds (Ax <= b)
5. Equality constraints (Ax = b)

Variables in standard form:
1. Original variables (x1, x2, ...)
2. Slack variables for inequalities
3. Artificial variables for equalities

All slack variables are negative since all constraints are converted to >= form:
- Lower bounds (x >= b) become x - s = b
- Upper bounds (x <= b) become -x - s = -b (after converting to -x >= -b)
- Equalities (x = b) get artificial variables
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from boundhound.core.problem import LPProblem, StandardForm
from boundhound.types import LPSolution, PhaseType, SimplexStatus

T = TypeVar("T", bound=np.floating)

logger = logging.getLogger(__name__)


@dataclass
class RevisedSimplexSolver:
    """Revised simplex method solver for linear programs in standard form.

    The revised simplex method maintains an explicit basis inverse and
    uses it to compute reduced costs and update solutions efficiently.
    """

    def __init__(self, problem: StandardForm, tol: float = 1e-8):
        """Initialize solver with problem data and initial basis."""
        self.problem = problem
        self._basis = self.problem.basis.copy()
        self.tol = tol
        self.update_basis_inverse()

    def update_basis_inverse(self) -> None:
        """Update basis inverse after basis changes."""
        self._basis_inverse = np.linalg.inv(self.problem.A[:, self._basis.indices])

    @property
    def original_problem(self) -> LPProblem:
        """Get the original inequality form problem by traversing the parent chain.

        The parent chain can be:
        - Phase 2 -> Phase 1 -> LPProblem
        - Phase 2 -> LPProblem
        - Phase 1 -> LPProblem
        """
        current = self.problem
        while current.parent is not None and not isinstance(current.parent, LPProblem):
            current = current.parent
        if not isinstance(current.parent, LPProblem):
            raise ValueError("Could not find original inequality form problem in parent chain")
        return current.parent

    @property
    def basic_solution(self) -> NDArray[T]:
        """Current basic solution x_B = B^{-1}b."""
        x_B = self._basis_inverse @ self.problem.b
        return x_B

    @property
    def simplex_multipliers(self) -> NDArray[T]:
        """Simplex multipliers π = c_B^T B^{-1}."""
        c_B = self.problem.c[self._basis.indices]
        pi = c_B @ self._basis_inverse
        return pi

    @property
    def reduced_costs(self) -> NDArray[T]:
        """Reduced costs c̄ = c - π^T A."""
        pi = self.simplex_multipliers
        reduced = self.problem.c - (pi @ self.problem.A)
        return reduced

    @property
    def entering_variable(self) -> Optional[int]:
        """Select entering variable with most negative reduced cost (for minimization)."""
        eligible = np.where(self.reduced_costs < -self.tol)[0]
        if len(eligible) == 0:
            return None
        entering_idx = eligible[np.argmin(self.reduced_costs[eligible])]
        return entering_idx

    @property
    def leaving_variable(self) -> Optional[int]:
        """Select leaving variable using minimum ratio test.

        Following the revised simplex method:
        1. Compute direction d = B^{-1} A
        2. If d ≤ 0, problem is unbounded
        3. For d_i > 0, compute ratios x_i/d_i
        4. Select minimum ratio as leaving variable
        """
        d = self._basis_inverse @ self.problem.A[:, self.entering_variable]
        if np.all(d <= self.tol):
            logger.warning("All directions non-positive - problem is unbounded")
            return None

        # Compute ratios x_B[i] / d[i] for d[i] > 0
        ratios = np.full_like(d, np.inf)
        positive_d = d > self.tol

        # Only compute ratios where d[i] > 0
        if not np.any(positive_d):
            return None

        # Select minimum ratio as leaving variable
        ratios[positive_d] = self.basic_solution[positive_d] / d[positive_d]
        leaving_idx = np.argmin(ratios)

        return leaving_idx

    @property
    def solution_vector(self) -> NDArray[T]:
        """Construct solution vector from basic solution.

        For Phase 1: Returns full solution vector including slack/artificial variables
        For Phase 2: Returns only original variables
        """
        # Construct full solution vector (including slack/artificial)
        x_full = np.zeros(len(self.problem.c))
        x_full[self._basis.indices] = self.basic_solution

        # For Phase 2, return only original variables
        if self.problem.phase_type == PhaseType.PHASE2:
            original_n = self.original_problem.n
            return x_full[:original_n]

        return x_full

    @property
    def obj_value(self) -> float:
        """Current value of objective."""
        fac = self.original_problem.obj_factor
        return fac * np.dot(self.problem.c[self._basis.indices], self.basic_solution)

    def solve(self, max_iter: int = 100) -> LPSolution:
        """Solve linear program using revised simplex method."""
        for _ in range(max_iter):
            # If no negative reduced costs and solution is feasible, we're optimal!
            if self.entering_variable is None:
                if np.all(self.basic_solution >= -self.tol):
                    logger.info("Found optimal solution: %s", self.solution_vector)
                    logger.info("Optimal objective value: %f", self.obj_value)
                    return LPSolution(
                        x=self.solution_vector,
                        value=self.obj_value,
                        status=SimplexStatus.OPTIMAL,
                        basis=self._basis,
                    )
                else:
                    # No negative reduced costs but solution is infeasible
                    logger.warning("No improving direction but solution is infeasible")
                    return LPSolution(
                        x=None, value=None, status=SimplexStatus.INFEASIBLE, basis=self._basis
                    )

            # Select entering and leaving variables
            if self.leaving_variable is None:
                logger.warning("Problem is unbounded")
                return LPSolution(
                    x=None, value=None, status=SimplexStatus.UNBOUNDED, basis=self._basis
                )

            # Perform pivot and update basis
            self._basis.indices[self.leaving_variable] = self.entering_variable
            self.update_basis_inverse()

        # Max iterations reached
        logger.warning("Maximum iterations (%d) reached", max_iter)

        return LPSolution(
            x=None, value=None, status=SimplexStatus.MAX_ITERATIONS, basis=self._basis
        )


def solve_phase1(
    phase1_problem: StandardForm, max_iter: int = 100, tol: float = 1e-8
) -> Tuple[bool, LPSolution]:
    """Solve Phase 1 to find initial feasible solution.

    Phase 1 minimizes sum of artificial variables to find feasible solution.
    A solution is feasible only if all artificial variables are zero (within tolerance).
    Returns (is_feasible, solution) where is_feasible is True if all artificial variables ≈ 0.
    """
    if phase1_problem.phase_type != PhaseType.PHASE1:
        raise ValueError("Problem not in phase1 form")

    # Create and solve Phase 1 problem, then heck if artificial variables are zero
    solution = RevisedSimplexSolver(problem=phase1_problem, tol=tol).solve(max_iter)
    is_feasible = solution.status == SimplexStatus.OPTIMAL

    return is_feasible, solution


def solve_phase2(
    phase2_problem: StandardForm,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> LPSolution:
    """Solve Phase 2 using initial feasible basis from Phase 1."""
    if phase2_problem.phase_type != PhaseType.PHASE2:
        raise ValueError("Problem not in phase2 form")

    return RevisedSimplexSolver(problem=phase2_problem, tol=tol).solve(max_iter)


def solve_lp(
    problem: Union[LPProblem, StandardForm], max_iter: int = 1000, tol: float = 1e-8
) -> LPSolution:
    """Solve linear program using two-phase revised simplex method.

    Phase 1: Find initial feasible solution
    - Adds artificial variables for equality/lower bound constraints
    - Minimizes sum of artificial variables
    - Solution is feasible if optimal value ≈ 0

    Phase 2: Solve original problem
    - Removes artificial variables
    - Restores original objective
    - Uses feasible basis from Phase 1
    """
    # Convert to standard form if needed
    std_problem = problem if isinstance(problem, StandardForm) else problem.to_standard_form()

    # If already in Phase 2, we can solve directly
    if std_problem.phase_type == PhaseType.PHASE2:
        logger.info("Problem already in Phase 2 - solving directly")
        return solve_phase2(std_problem, max_iter, tol)

    #  If not, we need to solve the Phase 1 problem first, and then proceed to Phase 2
    phase1_problem = std_problem
    feasible, phase1_solution = solve_phase1(phase1_problem, max_iter, tol)
    if not feasible:
        logger.warning("Problem is infeasible")
        return LPSolution(
            x=None, value=None, status=SimplexStatus.INFEASIBLE, basis=phase1_solution.basis
        )

    # Convert Phase 1 solution to Phase 2 and solve
    logger.info("Converting to Phase 2")
    phase2_problem = phase1_problem.to_phase2_form(phase1_solution)

    return solve_phase2(phase2_problem, max_iter, tol)
