from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from boundhound.core.problem import LPProblem, MILPProblem
from boundhound.simplex import SimplexStatus, solve_lp
from boundhound.types import LPSolution, MILPStatus, NodeStatus, Sense
from boundhound.visualization import plot_tree

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """A node in the branch and bound tree."""

    problem: LPProblem  # The LP relaxation at this node
    parent: Optional[int]  # Index of parent node (None for root)
    depth: int  # Depth in the tree (0 for root)
    lower_bounds: Dict[int, float]  # Additional lower bounds on variables at this node
    upper_bounds: Dict[int, float]  # Additional upper bounds on variables at this node
    node_id: Optional[int] = None  # Order in which this node was processed
    solution: Optional[LPSolution] = None  # Solution to the LP relaxation
    status: Optional[NodeStatus] = None  # Status of this node
    branching_var: Optional[int] = None  # Variable we branched on
    branch_direction: Optional[str] = None  # 'left' (<=) or 'right' (>=)
    branch_value: Optional[float] = None  # Value we branched at
    best_integer_node: Optional[Node] = None  # Node that found the best integer solution

    @property
    def local_dual_bound(self) -> Optional[float]:
        """The best possible objective value for this subtree."""
        if self.solution is None or self.solution.status == SimplexStatus.INFEASIBLE:
            return float("-inf") if self.problem.sense == Sense.MAX else float("inf")
        return self.solution.value

    @property
    def mip_gap(self) -> Optional[float]:
        """Gap between this node's bound and best integer solution."""
        # Can't compute gap without a solution
        if self.solution is None or self.solution.status == SimplexStatus.INFEASIBLE:
            return None

        # Can't compute gap without a best integer solution
        if self.best_integer_node is None:
            return None

        # For optimal nodes, gap is 0 by definition
        if self.status == NodeStatus.OPTIMAL:
            return 0.0

        # For all other nodes, gap shows distance between bound and best integer
        gap = abs(self.local_dual_bound - self.best_integer_node.solution.value)
        denominator = max(abs(self.best_integer_node.solution.value), 1.0)

        return gap / denominator

    def __str__(self) -> str:
        """Format node information as a string."""
        # Format solution vector
        if self.solution and self.solution.x is not None:
            sol_str = ", ".join(f"x[{i}]={x:.2f}" for i, x in enumerate(self.solution.x))
        else:
            sol_str = "No solution found"

        # Format node's local dual bound (its LP objective)
        local_bound_str = (
            f"{self.local_dual_bound:.2f}" if self.local_dual_bound is not None else "N/A"
        )

        # Format MIP gap - same formula for all nodes
        gap_str = f"MIP Gap: {self.mip_gap:.2%}" if self.mip_gap is not None else "MIP Gap: N/A"

        # Format status
        status_str = self.status.name if self.status else "PENDING"

        # Format best integer solution found so far with node ID
        best_int_str = (
            f"Best Int: {self.best_integer_node.solution.value:.2f} (Node {self.best_integer_node.node_id})"
            if self.best_integer_node is not None
            else "Best Int: None"
        )

        # Add node ID to display
        node_id_str = f"Node: {self.node_id}" if self.node_id is not None else ""

        return (
            f"{node_id_str}\n"
            f"{sol_str}\n"
            f"Local Bound: {local_bound_str}\n"
            f"{best_int_str}\n"
            f"{gap_str}\n"
            f"Status: {status_str}"
        )


class MILPSolution(NamedTuple):
    """Solution of a mixed-integer linear program."""

    optimal_node_index: Optional[int]  # Index of the node with best integer solution
    status: MILPStatus  # Status indicating how the algorithm terminated
    nodes_processed: int  # Number of branch and bound nodes processed
    nodes_remaining: int  # Number of unexplored nodes remaining
    tree: List[Node]  # List of all nodes in the branch and bound tree

    @property
    def optimal_node(self) -> Optional[Node]:
        """The node containing the best integer solution (if any)."""
        if self.optimal_node_index is None:
            return None
        return self.tree[self.optimal_node_index]

    @property
    def x(self) -> Optional[np.ndarray]:
        """Solution vector (None if infeasible/unbounded)."""
        node = self.optimal_node
        return node.solution.x if node else None

    @property
    def value(self) -> Optional[float]:
        """Objective at solution (None if infeasible/unbounded)."""
        node = self.optimal_node
        return node.solution.value if node else None

    @property
    def mip_gap(self) -> Optional[float]:
        """Optimality gap between incumbent and best bound."""
        node = self.optimal_node
        return node.mip_gap if node else None

    def render(self, output_format: str = "png") -> None:
        """Visualize the branch and bound tree."""
        plot_tree(self.tree, output_format)


class BranchAndBoundSolver:
    """Solver for mixed-integer linear programs using branch and bound.

    This class implements the branch and bound algorithm for solving MILPs.
    It maintains the search tree and tracks the best solution found so far.

    Attributes:
        milp: The MILP to solve
        max_nodes: Maximum number of nodes to explore
        tol: Tolerance for considering a value integer
        active_nodes: Queue of unexplored nodes
        processed_nodes: List of processed nodes in order
        optimal_node: Node containing the best integer solution found
        nodes_processed: Number of nodes processed so far
    """

    def __init__(
        self,
        milp: MILPProblem,
        max_nodes: int = 100,
        tol: float = 1e-10,
    ):
        """Initialize the solver with a problem instance."""
        self.milp = milp
        self.max_nodes = max_nodes
        self.tol = tol

        # Initialize solver state
        self.active_nodes = []
        self.processed_nodes = []
        self.optimal_node = None
        self.nodes_processed = 0

    @property
    def is_maximization(self) -> bool:
        """Whether this is a maximization problem."""
        return self.milp.lp.sense == Sense.MAX

    @property
    def incumbent(self) -> Optional[np.ndarray]:
        """Current best integer solution found."""
        return self.optimal_node.solution.x.copy() if self.optimal_node else None

    def solve(self) -> MILPSolution:
        """Execute the branch and bound algorithm.

        Returns:
            MILPSolution containing the optimal solution (if found) and search tree.
        """
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
        solution = solve_lp(current.problem)
        current.solution = solution

        # Handle special cases first
        if solution.status == SimplexStatus.INFEASIBLE:
            self._handle_infeasible_node(current)
            return
        elif solution.status == SimplexStatus.UNBOUNDED:
            self._handle_unbounded_node(current)
            return

        # Process solution
        is_integer = self._is_integer_feasible(solution.x)
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

    def _is_integer_feasible(self, x: np.ndarray) -> bool:
        """Check if solution is integer feasible."""
        if x is None:
            return False
        return all(abs(x[i] - round(x[i])) <= self.tol for i in self.milp.integer_vars)

    def _handle_integer_solution(self, node: Node) -> None:
        """Process a node with an integer solution."""
        node.status = NodeStatus.INTEGER_FEASIBLE
        better_solution = self._is_better_solution(node)

        if better_solution:
            if self.optimal_node:
                self.optimal_node.status = NodeStatus.INTEGER_FEASIBLE
            self.optimal_node = node
            node.status = NodeStatus.OPTIMAL
            node.best_integer_node = node
        else:
            node.best_integer_node = self.optimal_node

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
        if self.optimal_node is None:
            return True

        if self.is_maximization:
            return node.solution.value > self.optimal_node.solution.value + self.tol
        return node.solution.value < self.optimal_node.solution.value - self.tol

    def _should_prune(self, node: Node) -> bool:
        """Determine if a node should be pruned."""
        if self.optimal_node is None:
            return False

        if self.is_maximization:
            return node.solution.value + self.tol <= self.optimal_node.solution.value
        return node.solution.value - self.tol >= self.optimal_node.solution.value

    def _branch_on_node(self, node: Node) -> None:
        """Create and queue child nodes by branching."""
        _, branch_var = self._find_branching_variable(node.solution.x)
        left, right = self._create_child_nodes(node, branch_var)
        self.active_nodes.extend([right, left])

    def _find_branching_variable(self, x: np.ndarray) -> Tuple[bool, Optional[int]]:
        """Find the most fractional variable to branch on."""
        frac_parts = {i: abs(x[i] - round(x[i])) for i in self.milp.integer_vars}
        is_integer = all(f <= self.tol for f in frac_parts.values())
        branch_var = None if is_integer else max(frac_parts, key=frac_parts.get)
        return is_integer, branch_var

    def _create_child_nodes(self, parent: Node, branch_var: int) -> Tuple[Node, Node]:
        """Create left and right child nodes for branching."""
        x = parent.solution.x
        floor_val = np.floor(x[branch_var])
        ceil_val = np.ceil(x[branch_var])

        # Left child: x[j] <= floor(x[j])
        left_problem = parent.problem.copy()
        left_problem.ub[branch_var] = min(left_problem.ub[branch_var], floor_val)
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
        right_problem.lb[branch_var] = max(right_problem.lb[branch_var], ceil_val)
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
        # Check if any node was unbounded
        if any(node.status == NodeStatus.UNBOUNDED for node in self.processed_nodes):
            return MILPStatus.UNBOUNDED

        # Check if we found an optimal solution
        if self.incumbent is not None:
            return MILPStatus.OPTIMAL if len(self.active_nodes) == 0 else MILPStatus.MAX_NODES

        # No solution found - either infeasible or hit max nodes
        return MILPStatus.INFEASIBLE if len(self.active_nodes) == 0 else MILPStatus.MAX_NODES


def solve_milp(milp: MILPProblem, max_nodes: int = 100, tol: float = 1e-10) -> MILPSolution:
    """Solve a mixed-integer linear program using branch and bound.

    Args:
        milp: The mixed-integer linear program to solve
        max_nodes: Maximum number of nodes to explore
        tol: Tolerance for considering a value integer

    Returns:
        MILPSolution containing the optimal solution (if found) and search tree
    """
    solver = BranchAndBoundSolver(milp, max_nodes, tol)
    return solver.solve()
