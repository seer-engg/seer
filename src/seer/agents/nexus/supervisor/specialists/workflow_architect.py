"""Workflow architect specialist agent."""
from typing import Any
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from seer.agents.nexus.supervisor.state import SupervisorState
from seer.agents.nexus.tools.workflow_tools import create_workflow_spec_structured
from seer.llm import get_llm_without_responses_api
from seer.logger import get_logger

logger = get_logger(__name__)


WORKFLOW_ARCHITECT_SYSTEM_PROMPT = """You are a workflow architect. Your job is to design complete workflow structures.

CAPABILITIES:
- get_workflow_template(query): Find pre-built templates to customize
- analyze_workflow(): Analyze existing workflow structure (for editing)

WORKFLOW STRUCTURE:
A workflow spec has:
- nodes: List of workflow blocks (tool, branch, transform, output)
- edges: Connections between nodes
- triggers: Optional list of triggers
- input_variables: Optional list of input variables

NODE TYPES:
1. tool: Execute a tool
   {
     "id": "fetch_data",
     "type": "tool",
     "tool": "tool_name",
     "inputs": {"param": "${trigger.data.field}"}
   }

2. llm: AI inference with structured output
   {
     "id": "analyze",
     "type": "llm",
     "inputs": {
       "model": "gpt-4o-mini",
       "prompt": "Analyze: ${fetch_data}",
       "data": "${fetch_data}"
     },
     "outputs": {
       "mode": "json",
       "schema": {"json_schema": {"type": "object", "properties": {"summary": {"type": "string"}}}}
     }
   }

3. task: Data transformation
   {
     "id": "extract",
     "type": "task",
     "kind": "set",
     "value": "${fetch_data.record}"
   }

4. if: Conditional branching
   {
     "id": "check_status",
     "type": "if",
     "condition": "${fetch_data.success} == true"
   }

5. for_each: Iterate over items
   {
     "id": "process_items",
     "type": "for_each",
     "items": "${fetch_data.items}",
     "item_var": "item",
     "index_var": "index"
   }

INSTRUCTIONS:
1. If discovered_tools or discovered_triggers are available, use them
2. Try to find a template first with get_workflow_template()
3. If template found, customize it based on user requirements
4. If no template, design from scratch
5. Create valid node IDs and proper data references (${...})
6. Connect nodes with edges

OUTPUT FORMAT:
Return a complete workflow spec as JSON:
{
    "version": "2",
    "triggers": [...],
    "nodes": [...],
    "edges": [...]
}

IMPORTANT:
- All tool names must match discovered tools exactly
- All trigger keys must match discovered triggers exactly
- Use ${trigger.data.field} for trigger data
- Use ${node_id} or ${node_id.field} for node outputs (node ID is the output variable)
- Edges connect nodes: {"source": "node1", "target": "node2"}
- Trigger edges: {"source": "trigger_id", "target": "node_id", "type": "trigger"}
"""


async def workflow_architect_specialist(state: SupervisorState) -> dict[str, Any]:
    """
    Workflow architect specialist agent.

    Designs complete workflow structure using structured output with automatic Pydantic validation.
    """
    llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)

    # Extract context from state
    user_intent = state.get("user_intent", "")
    discovered_tools = state.get("discovered_tools", [])
    discovered_triggers = state.get("discovered_triggers", [])

    # Use structured output to generate validated workflow spec
    try:
        proposal = create_workflow_spec_structured(
            llm=llm,
            user_intent=user_intent,
            discovered_tools=discovered_tools,
            discovered_triggers=discovered_triggers,
        )

        # Already validated by Pydantic!
        workflow_draft = proposal.spec.model_dump()

        return {
            "workflow_draft": workflow_draft,
            "messages": [AIMessage(content=f"{proposal.summary}\n\nReasoning: {proposal.reasoning}")]
        }

    except ValidationError as e:
        # Pydantic validation failed - LLM generated invalid spec
        error_msg = f"Generated invalid workflow spec: {e}"
        logger.error(error_msg)
        return {
            "workflow_draft": None,
            "messages": [AIMessage(content=error_msg)]
        }
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Catch all generation errors to return friendly message
        error_msg = f"Failed to generate workflow: {e}"
        logger.exception("Workflow generation failed")
        return {
            "workflow_draft": None,
            "messages": [AIMessage(content=error_msg)]
        }
