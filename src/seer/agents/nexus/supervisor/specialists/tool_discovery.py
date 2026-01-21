"""Tool discovery specialist agent."""
import json
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from seer.agents.nexus.supervisor.state import SupervisorState
from seer.agents.nexus.tools.discovery_tools import search_tools, list_available_tools
from seer.llm import get_llm_without_responses_api


TOOL_DISCOVERY_SYSTEM_PROMPT = """You are a tool discovery expert. Your job is to find the best tools for user requests.

CAPABILITIES:
- search_tools(query, reasoning, integration_filter): Search for tools matching a query
- list_available_tools(integration_type): List all available tools (optionally filtered by integration)

INSTRUCTIONS:
1. Analyze the user's request to understand what actions they need
2. Use search_tools to find relevant tools
3. Return tool names with confidence scores
4. If search fails, use list_available_tools to explore
5. Focus on finding the RIGHT tools, not implementing workflows

OUTPUT FORMAT:
Return a JSON object with discovered tools:
{
    "discovered_tools": [
        {"tool": "tool_name", "integration": "Gmail", "confidence": 95, "description": "..."},
        ...
    ],
    "recommendation": "Brief explanation of why these tools were chosen"
}

IMPORTANT:
- Never ask users for tool names - discover them transparently
- Use natural language queries (e.g., "create draft", "send email")
- Return at least 1-3 relevant tools if available
"""


async def tool_discovery_specialist(state: SupervisorState) -> dict[str, Any]:
    """
    Tool discovery specialist agent.

    Finds relevant tools based on user request.
    """
    llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)
    tools = [search_tools, list_available_tools]
    llm_with_tools = llm.bind_tools(tools)

    # Build conversation context
    messages = [SystemMessage(content=TOOL_DISCOVERY_SYSTEM_PROMPT)]

    # Add user's original request
    if state.get("user_intent"):
        messages.append(HumanMessage(content=f"User request: {state['user_intent']}"))

    # Add recent conversation
    messages.extend(state["messages"][-5:])  # Last 5 messages for context

    # Invoke specialist
    response = await llm_with_tools.ainvoke(messages)

    # Parse tool results if any
    discovered_tools = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_tools":
                result = await search_tools.ainvoke(tool_call["args"])
                try:
                    result_data = json.loads(result)
                    if result_data.get("top_match"):
                        discovered_tools.append(result_data["top_match"])
                    for alt in result_data.get("alternatives", [])[:2]:  # Top 2 alternatives
                        discovered_tools.append(alt)
                except (json.JSONDecodeError, KeyError):
                    pass

    return {
        "discovered_tools": discovered_tools or None,
        "messages": [response]
    }
