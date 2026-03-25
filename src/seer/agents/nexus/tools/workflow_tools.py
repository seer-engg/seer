"""
Workflow agent tools for analysis and submitting complete workflow specs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from seer.agents.nexus.context import (
    _current_thread_id,
    get_user_for_thread,
)
from seer.agents.nexus.tracking import track_nexus_tool
from seer.agents.nexus.schema_context import (
    generate_node_type_reference,
    generate_validation_checklist_from_model,
    generate_trigger_reference,
    generate_edge_reference,
)
from seer.logger import get_logger
from seer.core.schema.models import WorkflowSpec

logger = get_logger(__name__)


# -----------------------------
# Structured Output Models
# -----------------------------


class WorkflowProposal(BaseModel):
    """
    Structured workflow proposal with validated spec and metadata.

    This model wraps WorkflowSpec to add explanation fields while
    maintaining full Pydantic validation of the spec itself.
    """
    spec: WorkflowSpec = Field(
        ...,
        description="Complete workflow specification conforming to v2 schema"
    )
    summary: str = Field(
        ...,
        max_length=512,
        description="1-2 sentence summary of what the workflow does"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of design decisions: why these tools/triggers, why this structure"
    )


def create_workflow_spec_structured(
    llm: BaseChatModel,
    user_intent: str,
    discovered_tools: List[Dict[str, Any]],
    discovered_triggers: List[Dict[str, Any]],
    template_hint: Optional[str] = None,
) -> WorkflowProposal:
    """
    Generate WorkflowSpec using structured output with automatic Pydantic validation.

    Uses LangChain's with_structured_output() to have the LLM generate a validated
    WorkflowSpec object. Pydantic validates:
    - Node types via discriminated union
    - Required fields
    - Field types
    - Nested structures

    Args:
        llm: Language model to use for generation
        user_intent: User's description of desired workflow
        discovered_tools: List of tools from search_tools() results
        discovered_triggers: List of triggers from search_triggers() results
        template_hint: Optional template name to use as starting point

    Returns:
        WorkflowProposal with validated spec, summary, and reasoning

    Raises:
        ValidationError: If generated spec fails Pydantic validation
    """
    structured_llm = llm.with_structured_output(WorkflowProposal, method="function_calling")

    # Format discovered tools/triggers for prompt
    tools_text = json.dumps(discovered_tools, indent=2) if discovered_tools else "None discovered"
    triggers_text = json.dumps(discovered_triggers, indent=2) if discovered_triggers else "None discovered"

    # Build prompt with auto-generated schema docs
    node_reference = generate_node_type_reference()
    validation_checklist = generate_validation_checklist_from_model()
    trigger_reference = generate_trigger_reference()
    edge_reference = generate_edge_reference()

    template_section = ""
    if template_hint:
        template_section = f"\n**Template Hint:** Consider using or adapting the '{template_hint}' template as a starting point.\n"

    prompt = f"""Design a complete workflow for: {user_intent}

Available tools:
{tools_text}

Available triggers:
{triggers_text}
{template_section}
**Node Types**
{node_reference}

**Triggers**
{trigger_reference}

**Edges**
{edge_reference}

**Requirements**
{validation_checklist}

Create a complete WorkflowSpec with:
- Appropriate nodes using discovered tools/triggers
- Edges connecting all nodes (no orphaned nodes)
- Variable references using ${{node_id}} or ${{node_id.field}} syntax
- Triggers with all required fields if workflow is trigger-based
- Proper edge types (default, trigger, conditional_true/false, loop_body/exit)

Provide a summary and reasoning for your design decisions."""

    # LLM generates WorkflowProposal, Pydantic validates automatically
    proposal = structured_llm.invoke([HumanMessage(content=prompt)])

    # If we get here, spec is valid!
    return proposal


def create_bound_get_workflow(workflow_id: str):
    """
    Create a get_workflow tool with workflow_id pre-bound.

    The returned tool fetches the workflow spec from the database using
    the pre-bound workflow_id, so the agent doesn't need to pass it.
    """
    @tool
    @track_nexus_tool("get_workflow")
    async def get_workflow() -> str:
        """
        Get the current workflow specification.

        Returns the full workflow including nodes, edges, and triggers as JSON.
        """
        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import get_workflow as get_workflow_service

        thread_id = _current_thread_id.get()
        user = await get_user_for_thread(thread_id) if thread_id else None
        if not user:
            return json.dumps({"error": "User context not available"})

        try:
            response = await get_workflow_service(user, workflow_id)
            return json.dumps({
                "workflow_id": response.workflow_id,
                "name": response.name,
                "spec": response.spec.model_dump(mode="json"),
            }, indent=2)
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly error
            logger.exception("Error getting workflow: %s", e)
            return json.dumps({"error": "not_found", "message": str(e)})

    return get_workflow


def create_bound_analyze_workflow(workflow_id: str):
    """
    Create an analyze_workflow tool with workflow_id pre-bound.

    The returned tool analyzes the workflow structure using the pre-bound
    workflow_id, so the agent doesn't need to pass it.
    """
    @tool
    @track_nexus_tool("analyze_workflow")
    async def analyze_workflow() -> str:
        """
        Analyze the current workflow structure.

        Returns block types, connections, triggers, and configuration breakdown as JSON.
        """
        # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
        from seer.api.workflows.services.lifecycle import get_workflow as get_workflow_service

        thread_id = _current_thread_id.get()
        user = await get_user_for_thread(thread_id) if thread_id else None
        if not user:
            return json.dumps({"error": "User context not available"})

        try:
            response = await get_workflow_service(user, workflow_id)
            # Use shared analysis helper
            from seer.services.workflows.analysis import build_workflow_analysis  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
            analysis = build_workflow_analysis(workflow_id, response.name, response.spec)
            return json.dumps(analysis, indent=2)

        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly error
            logger.exception("Error analyzing workflow: %s", e)
            return json.dumps({"error": "analysis_failed", "message": str(e)})

    return analyze_workflow


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


def _summarize_spec(spec: Dict[str, Any]) -> str:
    """Produce a short human summary for a WorkflowSpec."""
    if not spec:
        return "Workflow proposal"
    nodes = spec.get("nodes") or []
    node_types = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type", "node")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    if not node_types:
        return f"{len(nodes)} nodes"
    parts = [f"{count} {node_type}" for node_type, count in node_types.items()]
    return ", ".join(parts)


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



@tool
@track_nexus_tool("submit_workflow_spec")
async def submit_workflow_spec(  # pylint: disable=too-many-return-statements # Reason: Validation chain requires early returns for each validation step
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
    # Import here to avoid circular dependency at module load time
    from seer.database.workflow_models import WorkflowChatSession  # pylint: disable=import-outside-toplevel # Reason: Avoid circular dependency
    from seer.database.workflow_models import WorkflowProposal as WorkflowProposalModel  # pylint: disable=import-outside-toplevel # Reason: Avoid circular dependency

    # Nexus-specific pre-checks
    if error := _validate_thread_context():
        return error

    spec_dict, error = _validate_spec_format(workflow_spec)
    if error:
        return error

    # Resolve user for compilation context
    thread_id = _current_thread_id.get()
    user = await get_user_for_thread(thread_id)
    if not user:
        return _error_response("internal", "User context not available for compilation validation")

    # Run golden validation pipeline (schema, references, trigger fix, compilation, re-parse)
    from seer.tools.workflow_validation import run_full_validation  # pylint: disable=import-outside-toplevel # Reason: Avoid circular dependency
    validation = await run_full_validation(user, spec_dict)

    if not validation.success:
        # Kill #6: Add progressive hints on repeated failures
        hint = validation.error.hint or ""
        hint += "\nCommon fix: check field names against get_workflow_schema examples."
        return _error_response(
            validation.error.error_type,
            validation.error.message,
            hint.strip(),
        )

    # Kill #5: Warn if agent nodes used alongside primitive alternatives
    warnings = []
    spec_payload = validation.validated_spec.model_dump(mode="json")
    nodes = spec_payload.get("nodes", [])
    node_types = {n.get("type") for n in nodes if isinstance(n, dict)}
    has_agent = "agent" in node_types
    has_primitives = node_types & {"for_each", "hitl", "tool"}
    if has_agent and has_primitives:
        warnings.append(
            "Agent nodes cost 10x more than primitive nodes. "
            "Consider replacing agent nodes with for_each/hitl/tool nodes."
        )

    # Get session and workflow for this thread
    session = await WorkflowChatSession.get_or_none(thread_id=thread_id).prefetch_related('workflow', 'user')
    if not session:
        return _error_response("internal", "Chat session not found for current thread")

    # Create WorkflowProposal record
    proposal = await WorkflowProposalModel.create(
        workflow=session.workflow,
        session=session,
        created_by=user,
        summary=(summary or _summarize_spec(spec_payload))[:512],
        spec=spec_payload,
        status=WorkflowProposalModel.STATUS_PENDING,
        thread_id=thread_id,
    )

    response = {
        "status": "ok",
        "message": "Workflow spec recorded for review",
        "workflow_spec": spec_payload,
        "proposal_id": proposal.id,
    }

    if validation.schema_fixes:
        response["auto_fixes"] = {
            "trigger_schemas_updated": validation.schema_fixes,
            "recommendation": (
                "Event schemas were auto-corrected to use canonical schemas from the registry. "
                "If you have expressions referencing trigger data (e.g., ${trigger.data.subject}), "
                "verify they use correct field paths from the 'available_fields' shown above."
            )
        }

    if summary:
        response["summary"] = summary
    if warnings:
        response["warnings"] = warnings

    return json.dumps(response, indent=2)
