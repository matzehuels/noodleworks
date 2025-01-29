"""Module for visualizing the branch and bound solution tree."""

from typing import TYPE_CHECKING

from graphviz import Digraph

from boundhound.branch_and_bound import NodeStatus

if TYPE_CHECKING:
    from boundhound.branch_and_bound import Node


def plot_tree(nodes: list["Node"], output_format: str = "png") -> None:
    """Plot the branch and bound tree using graphviz with LaTeX formatting."""
    dot = Digraph("Branch and Bound Tree", format=output_format)
    dot.attr(rankdir="TB")  # Top to bottom direction

    # Color scheme for different node statuses
    colors = {
        NodeStatus.OPTIMAL: "limegreen",  # Bright green for optimal solution
        NodeStatus.INTEGER_FEASIBLE: "palegreen",
        NodeStatus.FRACTIONAL: "lightblue",
        NodeStatus.INFEASIBLE: "lightgrey",
        NodeStatus.UNBOUNDED: "pink",
        NodeStatus.PRUNED_BY_BOUND: "lightyellow",
    }

    # Set default node attributes
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Helvetica",
        margin="0.2",
        width="0",
        height="0",
        fontsize="11",
    )

    # Add nodes
    for i, node in enumerate(nodes):
        # Add node with appropriate color
        color = colors.get(node.status, "white") if node.status else "white"
        dot.node(str(i), str(node), shape="box", style="filled,rounded", fillcolor=color)

        # Add edge from parent with branching information
        if node.parent is not None:
            edge_label = ""
            if node.branching_var is not None:
                var_name = f"x[{node.branching_var}]"
                if node.branch_direction == "left":
                    edge_label = f"{var_name} ≤ {node.branch_value}"
                else:
                    edge_label = f"{var_name} ≥ {node.branch_value}"
            dot.edge(str(node.parent), str(i), label=edge_label)

    # Render the graph
    dot.render("branch_and_bound_tree", view=True, cleanup=True)
