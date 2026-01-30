"""
Stage 4 — Lower the validated WorkflowSpec into an execution plan.

V2 uses explicit edges to define control flow. The execution plan includes
edges and precomputed indices for efficient graph traversal.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from seer.core.schema.models import Edge, EdgeType, ForEachNode, IfNode, Node, SwitchCase, SwitchNode, WorkflowSpec

logger = logging.getLogger(__name__)


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


def _find_loop_body_nodes(
    loop_node_id: str,
    body_entry_id: str,
    outgoing_edges: Dict[str, List[Edge]],
    node_map: Dict[str, Node]
) -> Tuple[Set[str], Set[str]]:
    """
    Detect all nodes in the loop body and identify terminal nodes.

    Starting from the body entry node, follow default edges until we reach:
    - A node with an edge back to the loop node (already handled)
    - A node with no outgoing default edges (terminal)
    - Another control flow node (IfNode/ForEachNode) - don't traverse into it

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

    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        body_nodes.add(current_id)

        edges_out = outgoing_edges.get(current_id, [])
        default_edges = [e for e in edges_out if e.type == EdgeType.default]

        # Check for edge back to loop
        has_loop_back = any(e.target == loop_node_id for e in default_edges)
        if has_loop_back:
            # Already has explicit back-edge, don't mark as terminal
            continue

        # Check if this is a terminal node (no outgoing default edges)
        if not default_edges:
            terminal_nodes.add(current_id)
            continue

        # Add next nodes to queue, but stop at nested control flow
        for edge in default_edges:
            target_node = node_map.get(edge.target)
            if not target_node:
                continue

            # Stop at nested control flow nodes (don't traverse into them)
            if isinstance(target_node, (ForEachNode, IfNode)):
                terminal_nodes.add(current_id)
                continue

            queue.append(edge.target)

    return body_nodes, terminal_nodes


def _convert_if_to_switch(if_node: IfNode, edges: List[Edge]) -> Tuple[SwitchNode, List[Edge]]:
    """
    Convert deprecated IfNode to SwitchNode for unified execution.

    Internal transformation applied during compilation for backward compatibility.

    Args:
        if_node: The IfNode to convert
        edges: All edges in the workflow

    Returns:
        Tuple of (converted SwitchNode, updated edges)
    """
    logger.info("Converting deprecated IfNode %s to SwitchNode", if_node.id)

    # Create single case for the if condition
    switch_cases = [
        SwitchCase(condition=if_node.condition, label="__if_true")
    ]

    # Create SwitchNode with same ID and UI metadata
    switch_node = SwitchNode(
        id=if_node.id,
        type="switch",
        cases=switch_cases,
        ui=if_node.ui
    )

    # Convert edges
    new_edges = []
    for edge in edges:
        if edge.source == if_node.id:
            if edge.type == EdgeType.conditional_true:
                # True branch becomes switch_case with route="__if_true"
                new_edges.append(Edge(
                    source=edge.source,
                    target=edge.target,
                    type=EdgeType.switch_case,
                    route="__if_true",
                    ui=edge.ui
                ))
            elif edge.type == EdgeType.conditional_false:
                # False branch becomes switch_default
                new_edges.append(Edge(
                    source=edge.source,
                    target=edge.target,
                    type=EdgeType.switch_default,
                    ui=edge.ui
                ))
            else:
                new_edges.append(edge)
        else:
            new_edges.append(edge)

    return switch_node, new_edges


def build_execution_plan(spec: WorkflowSpec) -> ExecutionPlan:  # pylint: disable=too-complex  # Reason: control-flow lowering requires multiple phases.
    """
    Build an execution plan from the workflow spec.

    Computes entry node, edge indices, trigger routing, and loop body detection
    for efficient graph traversal.

    Deprecated IfNodes are automatically converted to SwitchNodes for unified execution.
    """
    # Convert deprecated IfNodes to SwitchNodes
    nodes = list(spec.nodes)
    edges = list(spec.edges)

    converted_nodes = []
    for node in nodes:
        if isinstance(node, IfNode):
            switch_node, edges = _convert_if_to_switch(node, edges)
            converted_nodes.append(switch_node)
        else:
            converted_nodes.append(node)

    # Use converted nodes for the rest of the execution plan
    nodes = converted_nodes

    # Build edge indices
    outgoing: Dict[str, List[Edge]] = defaultdict(list)
    incoming: Dict[str, List[Edge]] = defaultdict(list)
    trigger_targets: Dict[str, str] = {}

    for edge in edges:
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
        for node in nodes:
            if not incoming.get(node.id):
                entry_node_id = node.id
                break

    # Build node map for loop body detection
    node_map: Dict[str, Node] = {node.id: node for node in nodes}

    # Detect loop bodies for all ForEachNodes
    loop_body_nodes: Dict[str, Set[str]] = {}
    loop_terminal_nodes: Dict[str, Set[str]] = {}

    for node in nodes:
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
        nodes=nodes,
        edges=edges,
        entry_node_id=entry_node_id,
        outgoing_edges=dict(outgoing),
        incoming_edges=dict(incoming),
        trigger_targets=trigger_targets,
        loop_body_nodes=loop_body_nodes,
        loop_terminal_nodes=loop_terminal_nodes,
    )
