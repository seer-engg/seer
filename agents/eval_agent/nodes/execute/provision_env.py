"""Provision environment for a single dataset example based on natural language instructions."""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.tool_service import get_tool_service
from shared.resource_utils import format_resource_hints
from shared.test_runner.action_executor import load_mcp_tools
from shared.parameter_population import (
    extract_all_context_variables,
    format_context_variables_for_llm,
)
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from shared.tools import canonicalize_tool_name

logger = get_logger("eval_agent.execute.provision")


from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig



@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return await handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )




SYSTEM_PROMPT = """
You are a helpful assistant that provisions the environment for the target agent. based on the instructions provided. you will use all the tools available to you to provision the environment.

# Important:
- Asana expects no offset parameter at all on the first page. Sending offset="" (empty string) is treated as an invalid pagination token, so Asana returns:
    offset: Your pagination token is invalid.
"""
USER_PROMPT = """
Provision the environment for the target agent.

Instructions:
{instructions}

Resources:
{resources}

Context:
{context}

github repo buggy-coder already exist in the organization seer-engg.
"""


async def provision_environment_node(state: TestExecutionState) -> dict:
    """Plan and execute provisioning steps for the test example."""
    # Set start time if not set
    started_at = datetime.utcnow()

    example = state.dataset_example
    instructions: List[str] =  example.expected_output.provision_environment

    # If no MCP services or no instructions, skip
    if not state.context.mcp_services or not instructions:
        logger.error("No provisioning instructions or MCP services; skipping provisioning phase.")
        return {}

    # Ensure tools initialized (for execution) and entries loaded (for parameter completion)
    tool_service = get_tool_service()
    await tool_service.initialize(state.context.mcp_services)
    tools_dict = await load_mcp_tools(state.context.mcp_services)
    selected_tools = state.tool_selection_log.selected_tools

    # Prepare prompt context
    context_vars = extract_all_context_variables(
        user_context=state.context.user_context,
        github_context=state.context.github_context,
        mcp_resources=state.mcp_resources,
    )
    formatted_context_vars = format_context_variables_for_llm(context_vars)
    resource_hints = format_resource_hints(state.mcp_resources)

    actual_tools = [tools_dict[canonicalize_tool_name(tool)] for tool in selected_tools]
    llm = ChatOpenAI(
        model="gpt-5",
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "medium"},
    )
    provisioning_agent = create_agent(
        model=llm,
        tools=actual_tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[handle_tool_errors]
    )
    user_prompt = HumanMessage(content=USER_PROMPT.format(instructions=instructions, resources=resource_hints, context=formatted_context_vars))

    result = await provisioning_agent.ainvoke(input={"messages": [user_prompt]}, config=RunnableConfig(recursion_limit=75))

    return {
        "mcp_resources": state.mcp_resources,
        "started_at": started_at,
    }


