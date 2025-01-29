from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from boundhound.core.problem import LPProblem, MILPProblem
from boundhound.simplex import SimplexStatus, solve_lp
from boundhound.types import LPSolution, MILPStatus, NodeStatus, Sense
from boundhound.visualization import plot_tree

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """A node in the branch and bound tree."""

    problem: LPProblem
    parent: int | None
    depth: int
    lower_bounds: dict[int, float]
    upper_bounds: dict[int, float]
    node_id: int | None = None
    solution: LPSolution | None = None
    status: NodeStatus | None = None
    branching_var: int | None = None
    branch_direction: str | None = None
    branch_value: float | None = None
    best_integer_node: Node | None = None

    @property
    def local_dual_bound(self) -> float:
        """The best possible objective value for this subtree."""
        if self.solution is None or self.solution.status == SimplexStatus.INFEASIBLE:
            return float("-inf") if self.problem.sense == Sense.MAX else float("inf")
        return self.solution.value or float("inf")

    @property
    def mip_gap(self) -> float | None:
        """Gap between this node's bound and best integer solution."""
        # Can't compute gap without a valid solution
        if (self.solution is None) or (self.solution.status == SimplexStatus.INFEASIBLE):
            return None

        # Can't compute gap without a best integer solution
        if (
            self.best_integer_node is None
            or self.best_integer_node.solution is None
            or self.best_integer_node.solution.value is None
        ):
            return None

        # For optimal nodes, gap is 0 by definition
        if self.status == NodeStatus.OPTIMAL:
            return 0.0

        # For all other nodes, gap shows distance between bound and best integer
        best_value = self.best_integer_node.solution.value
        gap = abs(self.local_dual_bound - best_value)
        denominator = max(abs(best_value), 1.0)

        return gap / denominator

    def __str__(self) -> str:
        """Format node information as a string."""
        parts = []

        # Add node ID if available
        if self.node_id is not None:
            parts.append(f"Node: {self.node_id}")

        # Format solution vector if available
        if self.solution and self.solution.x is not None:
            sol_str = ", ".join(f"x[{i}]={x:.2f}" for i, x in enumerate(self.solution.x))
            parts.append(sol_str)
        else:
            parts.append("No solution found")

        # Add local bound
        parts.append(f"Local Bound: {self.local_dual_bound:.2f}")

        # Add best integer solution if available
        if (
            self.best_integer_node is not None
            and self.best_integer_node.solution is not None
            and self.best_integer_node.solution.value is not None
        ):
            parts.append(
                f"Best Int: {self.best_integer_node.solution.value:.2f} "
                f"(Node {self.best_integer_node.node_id})"
            )
        else:
            parts.append("Best Int: None")

        # Add MIP gap if available
        gap = self.mip_gap
        parts.append(f"MIP Gap: {gap:.2%}" if gap is not None else "MIP Gap: N/A")

        # Add status
        parts.append(f"Status: {self.status.name if self.status else 'PENDING'}")

        return "\n".join(parts)


class MILPSolution(NamedTuple):
    """Solution of a mixed-integer linear program."""

    optimal_node_index: int | None
    status: MILPStatus
    nodes_processed: int
    nodes_remaining: int
    tree: list[Node]

    @property
    def optimal_node(self) -> Node | None:
        """The node containing the best integer solution (if any)."""
        if self.optimal_node_index is None:
            return None
        return self.tree[self.optimal_node_index]

    @property
    def x(self) -> NDArray[np.float64] | None:
        """Solution vector (None if infeasible/unbounded)."""
        node = self.optimal_node
        if node is None or node.solution is None:
            return None
        return node.solution.x

    @property
    def value(self) -> float | None:
        """Objective at solution (None if infeasible/unbounded)."""
        node = self.optimal_node
        if node is None or node.solution is None:
            return None
        return node.solution.value

    @property
    def mip_gap(self) -> float | None:
        """Optimality gap between incumbent and best bound."""
        node = self.optimal_node
        return node.mip_gap if node is not None else None

    def render(self, output_format: str = "png") -> None:
        """Visualize the branch and bound tree."""
        plot_tree(self.tree, output_format)


class BranchAndBoundSolver:
    """Solver for mixed-integer linear programs using branch and bound.

    This class implements the branch and bound algorithm for solving MILPs.
    It maintains the search tree and tracks the best solution found so far.
    """

    def __init__(
        self,
        milp: MILPProblem,
        max_nodes: int = 100,
        tol: float = 1e-10,
    ) -> None:
        """Initialize the solver with a problem instance."""
        self.milp = milp
        self.max_nodes = max_nodes
        self.tol = tol

        # Initialize solver state
        self.active_nodes: list[Node] = []
        self.processed_nodes: list[Node] = []
        self.optimal_node: Node | None = None
        self.nodes_processed: int = 0

    @property
    def is_maximization(self) -> bool:
        """Whether this is a maximization problem."""
        return self.milp.lp.sense == Sense.MAX

    @property
    def incumbent(self) -> NDArray[np.float64] | None:
        """Current best integer solution found."""
        if self.optimal_node is None or self.optimal_node.solution is None:
            return None
        x = self.optimal_node.solution.x
        return x.copy() if x is not None else None

    def solve(self) -> MILPSolution:
        """Execute the branch and bound algorithm."""
        # Initialize root node
        root = Node(
            problem=self.milp.lp,
            parent=None,
            depth=0,
            lower_bounds={},
            upper_bounds={},
        )
        self.active_nodes = [root]

        # Main branch and bound loop
        while self.active_nodes and self.nodes_processed < self.max_nodes:
            self._process_next_node()

        # Return solution with final status
        status = self._determine_final_status()
        return MILPSolution(
            optimal_node_index=self.optimal_node.node_id if self.optimal_node else None,
            status=status,
            nodes_processed=self.nodes_processed,
            nodes_remaining=len(self.active_nodes),
            tree=self.processed_nodes,
        )

    def _process_next_node(self) -> None:
        """Process the next node in the queue."""
        self.nodes_processed += 1
        current = self.active_nodes.pop()

        # Assign node ID and store best solution known so far
        current.node_id = self.nodes_processed - 1
        current.best_integer_node = self.optimal_node

        # Solve LP relaxation
        current.solution = solve_lp(current.problem)

        # Handle special cases first
        if current.solution.status == SimplexStatus.INFEASIBLE:
            self._handle_infeasible_node(current)
            return
        elif current.solution.status == SimplexStatus.UNBOUNDED:
            self._handle_unbounded_node(current)
            return

        # Process solution
        is_integer = self._is_integer_feasible(current.solution.x)
        if is_integer:
            self._handle_integer_solution(current)
        else:
            self._handle_fractional_solution(current)

        # Check if we should prune
        if self._should_prune(current):
            self._handle_pruned_node(current)
            return

        # Branch if needed
        if not is_integer:
            self._branch_on_node(current)

        self.processed_nodes.append(current)

    def _handle_infeasible_node(self, node: Node) -> None:
        """Process an infeasible node."""
        node.status = NodeStatus.INFEASIBLE
        self.processed_nodes.append(node)

    def _handle_unbounded_node(self, node: Node) -> None:
        """Process an unbounded node."""
        node.status = NodeStatus.UNBOUNDED
        self.processed_nodes.append(node)
        # Clear active nodes since problem is unbounded
        self.active_nodes.clear()

    def _is_integer_feasible(self, x: NDArray[np.float64] | None) -> bool:
        """Check if solution is integer feasible."""
        if x is None:
            return False
        return all(abs(x[i] - round(x[i])) <= self.tol for i in self.milp.integer_vars)

    def _handle_integer_solution(self, node: Node) -> None:
        """Process a node with an integer solution."""
        # First mark as integer feasible
        node.status = NodeStatus.INTEGER_FEASIBLE

        # Check if this is a better solution
        if not self._is_better_solution(node):
            node.best_integer_node = self.optimal_node
            return

        # Update previous optimal node if it exists
        if self.optimal_node is not None:
            self.optimal_node.status = NodeStatus.INTEGER_FEASIBLE

        # Set this as new optimal node
        self.optimal_node = node
        node.status = NodeStatus.OPTIMAL
        node.best_integer_node = node

    def _handle_fractional_solution(self, node: Node) -> None:
        """Process a node with a fractional solution."""
        node.status = NodeStatus.FRACTIONAL
        node.best_integer_node = self.optimal_node

    def _handle_pruned_node(self, node: Node) -> None:
        """Process a pruned node."""
        node.status = NodeStatus.PRUNED_BY_BOUND
        node.best_integer_node = self.optimal_node
        self.processed_nodes.append(node)

    def _is_better_solution(self, node: Node) -> bool:
        """Check if node's solution is better than current best."""
        if (
            self.optimal_node is None
            or self.optimal_node.solution is None
            or self.optimal_node.solution.value is None
        ):
            return True

        if node.solution is None or node.solution.value is None:
            return False

        if self.is_maximization:
            return node.solution.value > self.optimal_node.solution.value + self.tol
        return node.solution.value < self.optimal_node.solution.value - self.tol

    def _should_prune(self, node: Node) -> bool:
        """Determine if a node should be pruned."""
        if (
            self.optimal_node is None
            or self.optimal_node.solution is None
            or self.optimal_node.solution.value is None
        ):
            return False

        if node.solution is None or node.solution.value is None:
            return False

        if self.is_maximization:
            return node.solution.value + self.tol <= self.optimal_node.solution.value
        return node.solution.value - self.tol >= self.optimal_node.solution.value

    def _branch_on_node(self, node: Node) -> None:
        """Create and queue child nodes by branching."""
        if node.solution is None or node.solution.x is None:
            raise ValueError("Can't branch on node without solution.")

        _, branch_var = self._find_branching_variable(node.solution.x)
        if branch_var is None:
            return

        left, right = self._create_child_nodes(node, branch_var)
        self.active_nodes.extend([right, left])

    def _find_branching_variable(self, x: NDArray[np.float64]) -> tuple[bool, int | None]:
        """Find the most fractional variable to branch on."""
        frac_parts = {i: abs(x[i] - round(x[i])) for i in self.milp.integer_vars}
        is_integer = all(f <= self.tol for f in frac_parts.values())
        if is_integer:
            return is_integer, None
        branch_var = max(frac_parts.keys(), key=lambda k: frac_parts[k])
        return is_integer, branch_var

    def _create_child_nodes(self, parent: Node, branch_var: int) -> tuple[Node, Node]:
        """Create left and right child nodes for branching."""
        if parent.solution is None or parent.solution.x is None:
            raise ValueError("No valid solution for parent node")

        floor_val = float(np.floor(parent.solution.x[branch_var]))
        ceil_val = float(np.ceil(parent.solution.x[branch_var]))

        # Left child: x[j] <= floor(x[j])
        left_problem = parent.problem.copy()
        left_problem.ub[branch_var] = min(float(left_problem.ub[branch_var]), floor_val)
        left = Node(
            problem=left_problem,
            parent=len(self.processed_nodes),
            depth=parent.depth + 1,
            lower_bounds=parent.lower_bounds.copy(),
            upper_bounds=parent.upper_bounds | {branch_var: floor_val},
            branching_var=branch_var,
            branch_direction="left",
            branch_value=floor_val,
        )

        # Right child: x[j] >= ceil(x[j])
        right_problem = parent.problem.copy()
        right_problem.lb[branch_var] = max(float(right_problem.lb[branch_var]), ceil_val)
        right = Node(
            problem=right_problem,
            parent=len(self.processed_nodes),
            depth=parent.depth + 1,
            lower_bounds=parent.lower_bounds | {branch_var: ceil_val},
            upper_bounds=parent.upper_bounds.copy(),
            branching_var=branch_var,
            branch_direction="right",
            branch_value=ceil_val,
        )

        return left, right

    def _determine_final_status(self) -> MILPStatus:
        """Determine the final status of the solve."""
        if any(node.status == NodeStatus.UNBOUNDED for node in self.processed_nodes):
            return MILPStatus.UNBOUNDED

        if self.incumbent is None:
            return MILPStatus.INFEASIBLE

        return MILPStatus.OPTIMAL if len(self.active_nodes) == 0 else MILPStatus.MAX_NODES


def solve_milp(milp: MILPProblem, max_nodes: int = 100, tol: float = 1e-10) -> MILPSolution:
    """Solve a mixed-integer linear program using branch and bound."""
    solver = BranchAndBoundSolver(milp, max_nodes, tol)
    return solver.solve()
