"""
Workflow analysis utilities shared across Nexus agent tools and MCP tools.
"""

from typing import Any, Dict, List

from seer.core.schema.models import WorkflowSpec


def build_workflow_analysis(
    workflow_id: str,
    workflow_name: str,
    spec: WorkflowSpec,
) -> Dict[str, Any]:
    """
    Build a structured analysis of a workflow specification.

    Args:
        workflow_id: Public workflow ID (e.g., "wf_abc123")
        workflow_name: Human-readable workflow name
        spec: The WorkflowSpec to analyze

    Returns:
        Dict containing:
        - workflow_id, workflow_name
        - total_blocks, total_connections
        - block_types: count by type
        - blocks: list of {id, type, config}
        - connections: list of {source, target, type}
        - triggers: list of {id, key} if present
    """
    nodes = spec.nodes
    edges = spec.edges or []

    analysis: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "total_blocks": len(nodes),
        "total_connections": len(edges),
        "block_types": {},
        "blocks": [],
        "connections": [],
    }

    # Analyze nodes
    block_types: Dict[str, int] = {}
    blocks: List[Dict[str, Any]] = []
    for node in nodes:
        block_type = node.type
        block_types[block_type] = block_types.get(block_type, 0) + 1
        blocks.append({
            "id": node.id,
            "type": block_type,
            "config": node.model_dump(mode="json", exclude={"id", "type"}),
        })
    analysis["block_types"] = block_types
    analysis["blocks"] = blocks

    # Analyze edges
    connections: List[Dict[str, Any]] = []
    for edge in edges:
        connections.append({
            "source": edge.source,
            "target": edge.target,
            "type": edge.type,
        })
    analysis["connections"] = connections

    # Include triggers if present
    if spec.triggers:
        analysis["triggers"] = [{"id": t.id, "key": t.key} for t in spec.triggers]

    return analysis
