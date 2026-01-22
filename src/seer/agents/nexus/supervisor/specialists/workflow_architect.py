"""Workflow architect specialist agent."""
import json
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from seer.agents.nexus.supervisor.state import SupervisorState
from seer.agents.nexus.tools.workflow_tools import analyze_workflow, get_workflow_template
from seer.llm import get_llm_without_responses_api


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

    Designs complete workflow structure based on discovered tools/triggers.
    """
    llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)
    tools = [get_workflow_template, analyze_workflow]
    llm_with_tools = llm.bind_tools(tools)

    # Build conversation context
    messages = [SystemMessage(content=WORKFLOW_ARCHITECT_SYSTEM_PROMPT)]

    # Add user's original request
    if state.get("user_intent"):
        messages.append(HumanMessage(content=f"User request: {state['user_intent']}"))

    # Add discovered tools if available
    if state.get("discovered_tools"):
        tools_summary = "\n".join([
            f"- {t['tool']} ({t.get('integration', 'unknown')}): {t.get('description', '')}"
            for t in state["discovered_tools"]
        ])
        messages.append(HumanMessage(content=f"Available tools:\n{tools_summary}"))

    # Add discovered triggers if available
    if state.get("discovered_triggers"):
        triggers_summary = "\n".join([
            f"- {t['key']} ({t.get('provider', 'unknown')}): {t.get('description', '')}"
            for t in state["discovered_triggers"]
        ])
        messages.append(HumanMessage(content=f"Available triggers:\n{triggers_summary}"))

    # Add recent conversation
    messages.extend(state["messages"][-5:])

    # Invoke specialist
    response = await llm_with_tools.ainvoke(messages)

    # Try to extract workflow draft from response
    workflow_draft = None
    if response.content:
        try:
            # Try to parse JSON from response
            draft = json.loads(response.content)
            if isinstance(draft, dict) and "nodes" in draft:
                workflow_draft = draft
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        "workflow_draft": workflow_draft,
        "messages": [response]
    }
