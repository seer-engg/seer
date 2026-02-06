"""
Stage 4 — Lower the validated WorkflowSpec into an execution plan.

V2 uses explicit edges to define control flow. The execution plan includes
edges and precomputed indices for efficient graph traversal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from seer.core.schema.models import Edge, EdgeType, ForEachNode, Node, WorkflowSpec


@dataclass(frozen=True)
class ExecutionPlan:  # pylint: disable=too-many-instance-attributes  # Reason: execution plan carries multiple indexed views.
    """
    Execution plan with explicit graph structure.

    Attributes:
        nodes: All nodes in the workflow
        edges: All edges in the workflow
        entry_node_id: ID of the first node to execute (no incoming edges), None if trigger routing is used
        outgoing_edges: Map from node_id to list of edges leaving that node
        incoming_edges: Map from node_id to list of edges entering that node
        trigger_targets: Map from trigger_id to target node_id for routing
        loop_body_nodes: Map from loop_id to set of node_ids in the loop body
        loop_terminal_nodes: Map from loop_id to set of node_ids that are terminal in the loop body
    """
    nodes: List[Node]
    edges: List[Edge]
    entry_node_id: Optional[str]
    outgoing_edges: Dict[str, List[Edge]] = field(default_factory=dict)
    incoming_edges: Dict[str, List[Edge]] = field(default_factory=dict)
    trigger_targets: Dict[str, str] = field(default_factory=dict)  # trigger_id -> node_id
    loop_body_nodes: Dict[str, Set[str]] = field(default_factory=dict)  # loop_id -> body_node_ids
    loop_terminal_nodes: Dict[str, Set[str]] = field(default_factory=dict)  # loop_id -> terminal_node_ids


# =============================================================================
# BUG FIX: Loop Body Node Detection (2024-02 RCA - Trace Key Collision)
# =============================================================================
# PROBLEM: In a for_each loop, nodes AFTER an IfNode were not being detected
#   as part of the loop body. This caused trace key collision:
#   - All iterations wrote to `_trace_log_sent_status` (same key)
#   - Instead of `_trace_log_sent_status_iter_0`, `_trace_log_sent_status_iter_1`
#
# ROOT CAUSE: The original traversal ONLY followed EdgeType.default edges.
#   When encountering an IfNode or nested ForEachNode, it would:
#   1. Add the control flow node to body_nodes
#   2. Mark it as terminal and STOP (because no `default` edges from IfNode)
#   3. Miss all nodes connected via conditional_true/conditional_false/loop_exit
#
# ORIGINAL (BUGGY) CODE:
#   default_edges = [e for e in edges_out if e.type == EdgeType.default]
#   if isinstance(target_node, (ForEachNode, IfNode)):
#       terminal_nodes.add(current_id)  # WRONG: Treated control flow as terminal
#       continue
#
# WHY NOT CAUGHT: Test cases only covered simple linear loops (A -> B -> C).
#   No tests for loops containing if/else branches or nested loops.
#
# FIX: Traverse ALL control flow edge types, not just `default`:
#   - conditional_true: Edges from IfNode to "then" branch
#   - conditional_false: Edges from IfNode to "else" branch
#   - loop_body: Edges from ForEachNode to iteration body
#   - loop_exit: Edges from ForEachNode to nodes after loop completes
# =============================================================================
def _find_loop_body_nodes(
    loop_node_id: str,
    body_entry_id: str,
    outgoing_edges: Dict[str, List[Edge]],
    node_map: Dict[str, Node]
) -> Tuple[Set[str], Set[str]]:
    """
    Detect all nodes in the loop body and identify terminal nodes.

    Starting from the body entry node, follow all edge types to find nodes
    that are part of the loop body. This includes:
    - Regular nodes connected by default edges
    - Nested control flow nodes (IfNode, ForEachNode) and their children
    - Nodes after nested control flow (connected via on_complete, true_branch, false_branch)

    Traversal stops when we reach:
    - A node with an edge back to the loop node (already handled)
    - A node with no outgoing edges (terminal)
    - A node outside the loop (not reachable without going through the loop node)

    Args:
        loop_node_id: ID of the ForEachNode
        body_entry_id: ID of the first node in the loop body
        outgoing_edges: Map from node_id to list of outgoing edges
        node_map: Map from node_id to Node object

    Returns:
        (body_node_ids, terminal_node_ids)
    """
    body_nodes: Set[str] = set()
    terminal_nodes: Set[str] = set()
    visited: Set[str] = set()
    queue: List[str] = [body_entry_id]

    # -------------------------------------------------------------------------
    # CRITICAL: Must include ALL edge types that represent loop body control flow
    # -------------------------------------------------------------------------
    # The original bug only traversed `default` edges, missing:
    #   - Nodes after if/else (connected via conditional_true/conditional_false)
    #   - Nodes after nested loops (connected via loop_exit)
    # -------------------------------------------------------------------------
    body_edge_types = {
        EdgeType.default,           # Regular sequential edges: A -> B
        EdgeType.conditional_true,  # From IfNode to "then" branch
        EdgeType.conditional_false, # From IfNode to "else" branch
        EdgeType.loop_body,         # From ForEachNode to iteration body
        EdgeType.loop_exit,         # From ForEachNode to nodes after loop
    }

    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        body_nodes.add(current_id)

        edges_out = outgoing_edges.get(current_id, [])

        # Get all edges that lead to body nodes (not just default edges)
        body_edges = [e for e in edges_out if e.type in body_edge_types]

        # Check for edge back to loop (on any edge type)
        has_loop_back = any(e.target == loop_node_id for e in body_edges)
        if has_loop_back:
            # Already has explicit back-edge, don't mark as terminal
            continue

        # Check if this is a terminal node (no outgoing body edges)
        if not body_edges:
            terminal_nodes.add(current_id)
            continue

        # Add all reachable nodes to queue
        for edge in body_edges:
            target_id = edge.target
            # Don't loop back to the parent loop node
            if target_id == loop_node_id:
                continue
            target_node = node_map.get(target_id)
            if not target_node:
                continue
            queue.append(target_id)

    return body_nodes, terminal_nodes


def build_execution_plan(spec: WorkflowSpec) -> ExecutionPlan:  # pylint: disable=too-complex  # Reason: control-flow lowering requires multiple phases.
    """
    Build an execution plan from the workflow spec.

    Computes entry node, edge indices, trigger routing, and loop body detection
    for efficient graph traversal.
    """
    # Build edge indices
    outgoing: Dict[str, List[Edge]] = defaultdict(list)
    incoming: Dict[str, List[Edge]] = defaultdict(list)
    trigger_targets: Dict[str, str] = {}

    for edge in spec.edges:
        if edge.type == EdgeType.trigger:
            # Trigger edge: source is trigger ID, target is node
            trigger_targets[edge.source] = edge.target
            incoming[edge.target].append(edge)
        else:
            # Regular edge: source and target are both nodes
            outgoing[edge.source].append(edge)
            incoming[edge.target].append(edge)

    # Entry node: None when we have trigger routing (routed via triggers)
    entry_node_id: Optional[str] = None
    if not trigger_targets:
        # Fallback: find node with no incoming edges
        for node in spec.nodes:
            if not incoming.get(node.id):
                entry_node_id = node.id
                break

    # Build node map for loop body detection
    node_map: Dict[str, Node] = {node.id: node for node in spec.nodes}

    # Detect loop bodies for all ForEachNodes
    loop_body_nodes: Dict[str, Set[str]] = {}
    loop_terminal_nodes: Dict[str, Set[str]] = {}

    for node in spec.nodes:
        if isinstance(node, ForEachNode):
            # Find the loop_body edge to get entry point
            loop_edges = outgoing.get(node.id, [])
            body_entry: Optional[str] = None

            for edge in loop_edges:
                if edge.type == EdgeType.loop_body:
                    body_entry = edge.target
                    break

            if body_entry:
                # Detect all nodes in the loop body and terminal nodes
                body_nodes, terminal_nodes = _find_loop_body_nodes(
                    node.id,
                    body_entry,
                    outgoing,
                    node_map
                )
                loop_body_nodes[node.id] = body_nodes
                loop_terminal_nodes[node.id] = terminal_nodes

    return ExecutionPlan(
        nodes=list(spec.nodes),
        edges=list(spec.edges),
        entry_node_id=entry_node_id,
        outgoing_edges=dict(outgoing),
        incoming_edges=dict(incoming),
        trigger_targets=trigger_targets,
        loop_body_nodes=loop_body_nodes,
        loop_terminal_nodes=loop_terminal_nodes,
    )
