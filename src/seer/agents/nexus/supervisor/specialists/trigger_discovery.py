"""Trigger discovery specialist agent."""
import json
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from seer.agents.nexus.supervisor.state import SupervisorState
from seer.agents.nexus.tools.discovery_tools import search_triggers, list_available_triggers
from seer.llm import get_llm_without_responses_api


TRIGGER_DISCOVERY_SYSTEM_PROMPT = """You are a trigger discovery expert. Your job is to find and configure triggers for workflows.

CAPABILITIES:
- search_triggers(query, reasoning, provider_filter): Search for triggers matching a query
- list_available_triggers(provider): List all available triggers (optionally filtered by provider)

INSTRUCTIONS:
1. Analyze the user's request to understand when/how the workflow should trigger
2. Search for triggers based on timing or event requirements
3. Return trigger configurations with metadata
4. If search fails, use list_available_triggers to explore

TRIGGER TYPES:
- Schedule: Cron-based time triggers (e.g., "every day at 9am")
- Webhook: External HTTP events
- Provider events: Gmail new email, Supabase new row, etc.

OUTPUT FORMAT:
Return a JSON object with discovered triggers:
{
    "discovered_triggers": [
        {"key": "trigger_key", "provider": "gmail", "description": "...", "config_schema": {...}},
        ...
    ],
    "recommendation": "Brief explanation of why these triggers were chosen"
}

IMPORTANT:
- Focus on WHEN the workflow runs, not WHAT it does
- Match trigger types to user's timing requirements
- Return config schema so architect can use it
"""


async def trigger_discovery_specialist(state: SupervisorState) -> dict[str, Any]:
    """
    Trigger discovery specialist agent.

    Finds relevant triggers based on user request.
    """
    llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)
    tools = [search_triggers, list_available_triggers]
    llm_with_tools = llm.bind_tools(tools)

    # Build conversation context
    messages = [SystemMessage(content=TRIGGER_DISCOVERY_SYSTEM_PROMPT)]

    # Add user's original request
    if state.get("user_intent"):
        messages.append(HumanMessage(content=f"User request: {state['user_intent']}"))

    # Add recent conversation
    messages.extend(state["messages"][-5:])

    # Invoke specialist
    response = await llm_with_tools.ainvoke(messages)

    # Parse tool results
    discovered_triggers = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_triggers":
                result = await search_triggers.ainvoke(tool_call["args"])
                try:
                    result_data = json.loads(result)
                    for trigger in result_data.get("triggers", [])[:3]:  # Top 3
                        discovered_triggers.append(trigger)
                except (json.JSONDecodeError, KeyError):
                    pass

    return {
        "discovered_triggers": discovered_triggers or None,
        "messages": [response]
    }
