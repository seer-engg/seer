"""Provision environment for a single dataset example based on natural language instructions."""
from datetime import datetime
from typing import List

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.resource_utils import format_resource_hints
from langchain_core.messages import HumanMessage

logger = get_logger("eval_agent.execute.provision")


from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from .utils import COMMMON_TOOL_INSTRUCTIONS
from .utils import get_tools, llm



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

""" + COMMMON_TOOL_INSTRUCTIONS
USER_PROMPT = """
Provision the environment for the target agent based on the instructions provided.
Resorces:
{resources}

Instructions:
{instructions}

"""


async def provision_environment_node(state: TestExecutionState) -> dict:
    """Plan and execute provisioning steps for the test example."""
    # Set start time if not set
    started_at = datetime.utcnow()

    example = state.dataset_example
    instructions: List[str] =  example.expected_output.create_test_data

    # If no MCP services or no instructions, skip
    if not state.context.mcp_services or not instructions:
        logger.error("No provisioning instructions or MCP services; skipping provisioning phase.")
        return {}
    
    resource_hints = format_resource_hints(state.mcp_resources)

    actual_tools = await get_tools(state)
    
    provisioning_agent = create_agent(
        model=llm,
        tools=actual_tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[handle_tool_errors]
    )
    user_prompt = HumanMessage(content=USER_PROMPT.format(instructions=instructions, resources=resource_hints))

    result = await provisioning_agent.ainvoke(input={"messages": [user_prompt]}, config=RunnableConfig(recursion_limit=75))

    provisioning_output = result.get('messages')[-1].content

    return {
        "mcp_resources": state.mcp_resources,
        "started_at": started_at,
        "provisioning_output": provisioning_output,
    }


