"""Supervisor router for multi-agent architecture."""
from typing import Literal, Any
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from seer.agents.nexus.supervisor.state import SupervisorState
from seer.llm import get_llm_without_responses_api


class SupervisorDecision(BaseModel):
    """Structured output for supervisor routing decisions."""
    next_agent: Literal["tool_discovery", "trigger_discovery", "workflow_architect", "validation", "FINISH"]
    reasoning: str


SUPERVISOR_SYSTEM_PROMPT = """You are a workflow design supervisor. Route user requests to appropriate specialists.

SPECIALISTS:
- tool_discovery: Finds tools for user actions (email, database, API calls, etc.)
- trigger_discovery: Finds triggers for scheduling/events (cron, webhooks, provider events)
- workflow_architect: Designs complete workflow structure (nodes, edges, data flow)
- validation: Validates and fixes workflow specs

ROUTING STRATEGY:

1. NEW WORKFLOW CREATION (typical flow):
   User: "Create draft when signup"
   → tool_discovery (find "create draft" tool)
   → trigger_discovery (find "signup" trigger)
   → workflow_architect (design workflow structure)
   → validation (validate spec)
   → FINISH

2. INFORMATION QUERIES:
   User: "What tools can send email?"
   → tool_discovery
   → FINISH (answer given, no workflow needed)

   User: "How do I trigger on schedule?"
   → trigger_discovery
   → FINISH

3. VALIDATION ERRORS:
   Validation failed with errors
   → workflow_architect (fix and regenerate)
   → validation (retry)

4. EXISTING WORKFLOW EDITING:
   User: "Change the trigger to daily"
   → trigger_discovery (find new trigger)
   → workflow_architect (modify workflow)
   → validation

CURRENT STATE ANALYSIS:
- discovered_tools: {discovered_tools_status}
- discovered_triggers: {discovered_triggers_status}
- workflow_draft: {workflow_draft_status}
- validation_result: {validation_result_status}
- workflow_complete: {workflow_complete}

DECISION RULES:
1. If workflow_complete=True → FINISH
2. If validation_result shows errors → workflow_architect (to fix)
3. If workflow_draft exists but not validated → validation
4. If tools/triggers discovered but no draft → workflow_architect
5. If user asks about tools/triggers only → tool_discovery or trigger_discovery then FINISH
6. If nothing discovered yet → tool_discovery or trigger_discovery based on query

IMPORTANT:
- Route based on current state and user intent
- Don't loop unnecessarily
- Use FINISH when task is complete or question is answered
"""


async def supervisor_router(state: SupervisorState) -> str:
    """
    Supervisor routing function.

    Analyzes current state and determines next specialist to invoke.
    Returns next agent name or "FINISH".
    """
    # Check if workflow is complete
    if state.get("workflow_complete"):
        return "FINISH"

    # Build status summary for prompt
    status = {
        "discovered_tools_status": "available" if state.get("discovered_tools") else "not discovered",
        "discovered_triggers_status": "available" if state.get("discovered_triggers") else "not discovered",
        "workflow_draft_status": "exists" if state.get("workflow_draft") else "not created",
        "validation_result_status": str(state.get("validation_result", {}).get("status", "not validated")),
        "workflow_complete": state.get("workflow_complete", False)
    }

    # Format system prompt with current state
    prompt = SUPERVISOR_SYSTEM_PROMPT.format(**status)

    # Build conversation summary for routing decision
    recent_messages = state["messages"][-3:] if state["messages"] else []
    conversation_context = "\n".join([
        f"{msg.__class__.__name__}: {msg.content[:200]}..."
        for msg in recent_messages
    ])

    llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)
    llm_with_structure = llm.with_structured_output(SupervisorDecision)

    decision = await llm_with_structure.ainvoke([
        SystemMessage(content=prompt),
        SystemMessage(content=f"Recent conversation:\n{conversation_context}")
    ])

    return decision.next_agent


def route_to_specialist(state: SupervisorState) -> dict[str, Any]:
    """
    Conditional routing based on supervisor decision.

    Used by LangGraph conditional edges.
    """
    import asyncio  # pylint: disable=import-outside-toplevel  # Required for synchronous LangGraph edge function wrapping async router
    return {"next": asyncio.run(supervisor_router(state))}
