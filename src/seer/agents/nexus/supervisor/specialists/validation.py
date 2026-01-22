"""Validation specialist agent."""
import json
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from seer.agents.nexus.supervisor.state import SupervisorState
from seer.agents.nexus.tools.workflow_tools import submit_workflow_spec
from seer.llm import get_llm_without_responses_api


VALIDATION_SYSTEM_PROMPT = """You are a validation expert. Your job is to validate workflow specifications and fix errors.

CAPABILITIES:
- submit_workflow_spec(spec, summary): Validate and record workflow spec

VALIDATION PROCESS:
1. Review the workflow draft from workflow_architect
2. Call submit_workflow_spec() with the draft
3. If validation succeeds, workflow is complete
4. If validation fails, analyze the error and suggest fixes

NOTE: The workflow draft has already been validated by Pydantic for schema correctness.
This validation focuses on:
1. Tool names match registry
2. Trigger keys match registry
3. Compilation succeeds (type checking, references)

COMMON VALIDATION ERRORS:
1. Tool names don't match registry: Use exact tool names from discovered_tools
2. Trigger keys don't match registry: Use exact keys from discovered_triggers
3. Type mismatches in compilation: Field names must match (e.g., 'thread_id' vs 'threadId')
4. Invalid variable references: Check ${...} syntax points to valid nodes

ERROR HANDLING:
When validation fails:
1. Parse the error message
2. Identify the root cause
3. Explain to user what went wrong
4. Suggest specific fixes
5. Don't retry immediately - let supervisor coordinate

OUTPUT:
When validation succeeds, return success status.
When validation fails, explain the error clearly.

IMPORTANT:
- Schema is already validated - focus on tool/trigger/compilation errors
- Only validate - don't redesign the workflow
- Be specific about what needs to be fixed
- Provide actionable error messages
"""


async def validation_specialist(state: SupervisorState) -> dict[str, Any]:
    """
    Validation specialist agent.

    Validates workflow specs and provides error feedback.
    """
    llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)
    tools = [submit_workflow_spec]
    llm_with_tools = llm.bind_tools(tools)

    # Build conversation context
    messages = [SystemMessage(content=VALIDATION_SYSTEM_PROMPT)]

    # Add workflow draft if available
    if state.get("workflow_draft"):
        draft_json = json.dumps(state["workflow_draft"], indent=2)
        messages.append(HumanMessage(content=f"Workflow draft to validate:\n```json\n{draft_json}\n```"))

    # Add recent conversation
    messages.extend(state["messages"][-5:])

    # Invoke specialist
    response = await llm_with_tools.ainvoke(messages)

    # Parse validation result if tool was called
    validation_result = None
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "submit_workflow_spec":
                result = await submit_workflow_spec.ainvoke(tool_call["args"])
                try:
                    validation_result = json.loads(result)
                except (json.JSONDecodeError, KeyError):
                    validation_result = {"status": "error", "message": result}

    return {
        "validation_result": validation_result,
        "workflow_complete": validation_result and validation_result.get("status") == "ok",
        "messages": [response]
    }
