"""Common types and enums used across the package."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from boundhound.core.basis import Basis


class Sense(Enum):
    """Sense for optimization."""

    MIN = "min"
    MAX = "max"


class PhaseType(Enum):
    """Type of phase in the simplex method."""

    NONE = 0  # Standard form, no phase
    PHASE1 = 1  # Phase 1 auxiliary problem
    PHASE2 = 2  # Phase 2 original objective


class SimplexStatus(Enum):
    """Termination status of the simplex algorithm."""

    OPTIMAL = 1  # Found optimal solution
    INFEASIBLE = 2  # Problem is infeasible
    UNBOUNDED = 3  # Problem is unbounded
    MAX_ITERATIONS = 4  # Hit iteration limit


class NodeStatus(Enum):
    """Status of a branch and bound node."""

    OPTIMAL = 1  # Node solved to optimality
    INFEASIBLE = 2  # Node is infeasible
    UNBOUNDED = 3  # Node is unbounded
    INTEGER_FEASIBLE = 4  # Node has an integer feasible solution
    FRACTIONAL = 5  # Node has a fractional solution
    PRUNED_BY_BOUND = 6  # Node pruned by bound (objective worse than incumbent)


class MILPStatus(Enum):
    """Status of the branch and bound algorithm."""

    OPTIMAL = 1  # Found proven optimal integer solution
    INFEASIBLE = 2  # Problem is infeasible
    UNBOUNDED = 3  # Problem is unbounded
    MAX_NODES = 4  # Hit node limit without proving optimality
    NUMERICAL_ISSUES = 5  # Encountered numerical issues


class LPSolution(NamedTuple):
    """Solution to a linear program."""

    x: np.ndarray | None  # Solution vector (None if no solution found)
    value: float | None  # Objective value (None if no solution found)
    status: SimplexStatus  # Status of the solution
    basis: Basis  # Basis for solution
