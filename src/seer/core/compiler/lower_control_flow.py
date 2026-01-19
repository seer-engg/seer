"""
Stage 4 — Lower the validated WorkflowSpec into an execution plan.

V2 uses explicit edges to define control flow. The execution plan includes
edges and precomputed indices for efficient graph traversal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from seer.core.schema.models import Edge, EdgeType, Node, WorkflowSpec


@dataclass(frozen=True)
class ExecutionPlan:
    """
    Execution plan with explicit graph structure.

    Attributes:
        nodes: All nodes in the workflow
        edges: All edges in the workflow
        entry_node_id: ID of the first node to execute (no incoming edges), None if trigger routing is used
        outgoing_edges: Map from node_id to list of edges leaving that node
        incoming_edges: Map from node_id to list of edges entering that node
        trigger_targets: Map from trigger_id to target node_id for routing
    """
    nodes: List[Node]
    edges: List[Edge]
    entry_node_id: Optional[str]
    outgoing_edges: Dict[str, List[Edge]] = field(default_factory=dict)
    incoming_edges: Dict[str, List[Edge]] = field(default_factory=dict)
    trigger_targets: Dict[str, str] = field(default_factory=dict)  # trigger_id -> node_id


def build_execution_plan(spec: WorkflowSpec) -> ExecutionPlan:
    """
    Build an execution plan from the workflow spec.

    Computes entry node, edge indices, and trigger routing for efficient graph traversal.
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

    return ExecutionPlan(
        nodes=list(spec.nodes),
        edges=list(spec.edges),
        entry_node_id=entry_node_id,
        outgoing_edges=dict(outgoing),
        incoming_edges=dict(incoming),
        trigger_targets=trigger_targets,
    )
