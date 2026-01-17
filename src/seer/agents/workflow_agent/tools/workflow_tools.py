"""
Workflow agent tools for analysis and submitting complete workflow specs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from seer.agents.workflow_agent.context import (
    _current_thread_id,
    get_workflow_state_for_thread,
    set_proposed_spec_for_thread,
    get_user_for_thread,
)
from seer.agents.workflow_agent.schema_context import get_workflow_templates
from seer.logger import get_logger
from seer.tools.base import get_tool
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.errors import (
    ValidationPhaseError,
    TypeEnvironmentError,
    WorkflowCompilerError,
)
from seer.core.registry.trigger_registry import trigger_registry
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton

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
            logger.warning("Failed to parse workflow_spec string: %s", exc)
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _error_response(error_type: str, message: str, hint: Optional[str] = None) -> str:
    """Create a standardized error response for workflow spec validation."""
    response = {"status": "error", "error_type": error_type, "message": message}
    if hint:
        response["hint"] = hint
    return json.dumps(response)


def _validate_thread_context() -> Optional[str]:
    """Validate thread context exists. Returns error message if invalid, None if valid."""
    thread_id = _current_thread_id.get()
    if not thread_id:
        return _error_response(
            "internal", "submit_workflow_spec requires an active thread_id context"
        )
    return None


def _validate_spec_format(workflow_spec: Any) -> tuple[Optional[Dict], Optional[str]]:
    """Validate spec format. Returns (spec_dict, error_message)."""
    spec_dict = _coerce_spec_payload(workflow_spec)
    if spec_dict is None:
        return None, _error_response(
            "parsing", "workflow_spec must be an object that follows the compiler schema"
        )
    return spec_dict, None


def _validate_pydantic(spec_dict: Dict) -> tuple[Optional[Any], Optional[str]]:
    """Run Pydantic validation. Returns (validated_spec, error_message)."""
    try:
        validated_spec = parse_workflow_spec(spec_dict)
        return validated_spec, None
    except ValidationPhaseError as exc:
        logger.warning("Workflow spec validation failed", exc_info=exc)
        return None, _error_response("parsing", f"Workflow spec validation failed: {exc}")


def _validate_tools_and_triggers(spec_dict: Dict) -> Optional[str]:
    """
    Validate that all referenced tools and triggers exist.
    Returns error message if validation fails, None if valid.
    """
    errors = []

    # Check that all tool nodes reference registered tools
    nodes = spec_dict.get("nodes", [])
    for node in nodes:
        if node.get("type") == "tool":
            tool_name = node.get("tool")
            if tool_name and not get_tool(tool_name):
                errors.append(
                    f"Tool '{tool_name}' not found. Use search_tools('{tool_name.split('_')[0]}') to find the correct tool name."
                )

    # Check that all triggers reference registered trigger keys
    triggers = spec_dict.get("triggers", [])
    for trigger in triggers:
        trigger_key = trigger.get("key")
        if trigger_key and not trigger_registry.maybe_get(trigger_key):
            available_triggers = [t.key for t in trigger_registry.all()]
            errors.append(
                f"Trigger '{trigger_key}' not found. Available triggers: {', '.join(available_triggers)}. "
                f"Use search_triggers() to find the correct trigger key."
            )

    if errors:
        return _error_response(
            "tool_trigger_validation",
            "Workflow references non-existent tools or triggers",
            "\n".join(errors)
        )

    return None


async def _validate_compilation(spec_dict: Dict) -> Optional[str]:
    """
    Run full compilation validation. Returns error message if invalid, None if valid.

    Validates type environment, references, and compilation.
    """
    thread_id = _current_thread_id.get()
    user = get_user_for_thread(thread_id)
    if not user:
        return _error_response("internal", "User context not available for compilation validation")

    try:
        compiler = WorkflowCompilerSingleton.instance()
        await compiler.compile(user, spec_dict, checkpointer=None)
        return None
    except TypeEnvironmentError as exc:
        logger.warning("Workflow type environment validation failed", exc_info=exc)
        return _error_response(
            "type_environment",
            f"Type validation failed: {exc}",
            "Check that output schemas match input expectations. "
            "Common issue: field name mismatches like 'threadId' vs 'thread_id'.",
        )
    except ValidationPhaseError as exc:
        logger.warning("Workflow reference validation failed", exc_info=exc)
        return _error_response(
            "validation",
            f"Validation failed: {exc}",
            "Check that all ${...} references point to valid variables."
        )
    except WorkflowCompilerError as exc:
        logger.warning("Workflow compilation failed", exc_info=exc)
        return _error_response("compilation", f"Compilation failed: {exc}")


@tool
async def submit_workflow_spec(
    workflow_spec: Any,
    summary: Optional[str] = None,
) -> str:
    """
    Validate and record a complete workflow specification produced by the agent.

    This tool performs full compilation validation including type checking, reference
    validation, and dependency checks. If validation fails, the error is returned so
    you can fix the spec and retry.

    Args:
        workflow_spec: Full workflow JSON object conforming to workflow_compiler schema.
                       Can be provided as a dict or a JSON string.
        summary: Optional natural language rationale for the proposal.
    """
    # Validation chain - each returns early if error
    if error := _validate_thread_context():
        return error

    spec_dict, error = _validate_spec_format(workflow_spec)
    if error:
        return error

    validated_spec, error = _validate_pydantic(spec_dict)
    if error:
        return error

    # Validate tools and triggers exist before compilation
    if error := _validate_tools_and_triggers(spec_dict):
        return error

    if error := await _validate_compilation(spec_dict):
        return error

    # All validations passed - record the spec
    thread_id = _current_thread_id.get()
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


@tool
async def get_workflow_template(
    query: str,
) -> str:
    """
    Retrieve a workflow template by name or tags to use as a starting point.

    Use this when you identify that the user's request matches a common pattern.
    For example, if user wants "create gmail draft when supabase signup", search
    for templates with tags like "supabase", "gmail", "welcome".

    Args:
        query: Template name or tag to search for (e.g., "supabase gmail", "welcome", "slack notification")

    Returns:
        JSON with matching template(s) including full spec that can be customized
    """
    try:
        templates = get_workflow_templates()
        query_lower = query.lower()

        matches = []
        for template in templates:
            name = template.get("name", "").lower()
            tags = [t.lower() for t in template.get("tags", [])]
            description = template.get("description", "").lower()

            # Match if query appears in name, tags, or description
            if (query_lower in name or
                any(query_lower in tag for tag in tags) or
                query_lower in description):
                matches.append(template)

        if not matches:
            available_templates = [
                {"name": t.get("name"), "tags": t.get("tags", [])}
                for t in templates
            ]
            return json.dumps({
                "query": query,
                "matches": [],
                "message": f"No templates found matching '{query}'",
                "available_templates": available_templates,
                "suggestion": "Try searching with integration names (gmail, supabase, slack) or action words (welcome, notification, report)"
            })

        # Return matches with full specs
        results = []
        for match in matches:
            results.append({
                "name": match.get("name"),
                "description": match.get("description"),
                "tags": match.get("tags"),
                "customization_guide": match.get("customization_guide"),
                "spec": match.get("spec")
            })

        return json.dumps({
            "query": query,
            "matches": results,
            "count": len(results),
            "message": f"Found {len(results)} template(s) matching '{query}'"
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all to return friendly JSON error
        logger.exception("Error retrieving template: %s", e)
        return json.dumps({
            "query": query,
            "matches": [],
            "error": str(e)
        })
