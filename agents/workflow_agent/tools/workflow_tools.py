"""
Workflow agent tools for analysis and submitting complete workflow specs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from agents.workflow_agent.context import (
    _current_thread_id,
    get_workflow_state_for_thread,
    set_proposed_spec_for_thread,
)
from shared.logger import get_logger
from workflow_compiler.compiler.parse import parse_workflow_spec
from workflow_compiler.errors import ValidationPhaseError

logger = get_logger(__name__)


def _resolve_workflow_state(
    workflow_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Use explicit workflow_state if provided otherwise fall back to thread context."""
    if workflow_state is not None:
        return workflow_state
    thread_id = _current_thread_id.get()
    if thread_id:
        return get_workflow_state_for_thread(thread_id)
    return None


@tool
async def analyze_workflow(
    workflow_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Analyze the current workflow structure.

    Returns a JSON string describing the workflow's blocks, connections, and configuration.
    """
    resolved_state = _resolve_workflow_state(workflow_state)
    if resolved_state is None:
        return json.dumps({"error": "Workflow state not available"})

    nodes = resolved_state.get("nodes", [])
    edges = resolved_state.get("edges", [])

    analysis = {
        "total_blocks": len(nodes),
        "total_connections": len(edges),
        "block_types": {},
        "blocks": [],
        "connections": [],
    }

    for node in nodes:
        block_type = node.get("type", "unknown")
        analysis["block_types"][block_type] = analysis["block_types"].get(block_type, 0) + 1
        analysis["blocks"].append(
            {
                "id": node.get("id"),
                "type": block_type,
                "label": node.get("data", {}).get("label", ""),
                "config": node.get("data", {}).get("config", {}),
            }
        )

    for edge in edges:
        analysis["connections"].append(
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "branch": edge.get("data", {}).get("branch"),
            }
        )

    if resolved_state.get("block_aliases"):
        analysis["block_aliases"] = resolved_state["block_aliases"]
    if resolved_state.get("template_reference_examples"):
        analysis["template_reference_examples"] = resolved_state["template_reference_examples"]
    if resolved_state.get("input_variables"):
        analysis["input_variables"] = resolved_state["input_variables"]

    return json.dumps(analysis, indent=2)


def _coerce_spec_payload(raw_spec: Any) -> Optional[Dict[str, Any]]:
    """Support both dict and JSON-string payloads from the model."""
    if isinstance(raw_spec, dict):
        return raw_spec
    if isinstance(raw_spec, str):
        try:
            parsed = json.loads(raw_spec)
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse workflow_spec string: {exc}")
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


@tool
async def submit_workflow_spec(
    workflow_spec: Any,
    summary: Optional[str] = None,
) -> str:
    """
    Validate and record a complete workflow specification produced by the agent.

    Args:
        workflow_spec: Full workflow JSON object conforming to workflow_compiler schema.
                       Can be provided as a dict or a JSON string.
        summary: Optional natural language rationale for the proposal.
    """

    thread_id = _current_thread_id.get()
    if not thread_id:
        return json.dumps(
            {
                "status": "error",
                "message": "submit_workflow_spec requires an active thread_id context",
            }
        )

    spec_dict = _coerce_spec_payload(workflow_spec)
    if spec_dict is None:
        return json.dumps(
            {
                "status": "error",
                "message": "workflow_spec must be an object that follows the compiler schema",
            }
        )

    try:
        validated_spec = parse_workflow_spec(spec_dict)
    except ValidationPhaseError as exc:
        logger.warning("Workflow spec validation failed", exc_info=exc)
        return json.dumps(
            {
                "status": "error",
                "message": f"Workflow spec validation failed: {exc}",
            }
        )

    spec_payload = validated_spec.model_dump(mode="json")
    proposal_context = {"spec": spec_payload}
    if summary:
        proposal_context["summary"] = summary
    set_proposed_spec_for_thread(thread_id, proposal_context)

    response = {
        "status": "ok",
        "message": "Workflow spec recorded for review",
        "workflow_spec": spec_payload,
    }
    if summary:
        response["summary"] = summary

    return json.dumps(response, indent=2)
