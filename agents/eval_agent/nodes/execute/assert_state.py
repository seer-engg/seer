"""Assert final state for a single dataset example based on natural language instructions."""
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.schema import ExperimentResultContext, FailureAnalysis
from shared.tool_service import get_tool_service
from shared.resource_utils import format_resource_hints
from shared.test_runner.action_executor import load_mcp_tools
from shared.parameter_population import (
    extract_all_context_variables,
    format_context_variables_for_llm,
)
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from shared.tools import canonicalize_tool_name
from shared.llm import convert_response_v1_output_to_message_string
from shared.tools import web_search


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



logger = get_logger("eval_agent.execute.assert")



SYSTEM_PROMPT = """
You are a helpful assistant that asserts the final state of the environment for the target agent. based on the specified criterias. you will use all the tools available to you to assert the final state.
"""
USER_PROMPT = """
Assert the following criterias by using the tools available to you:
{criterias}
"""


async def assert_final_state_node(state: TestExecutionState) -> dict:
    """Plan and execute assertion steps, then produce ExperimentResultContext."""
    example = state.dataset_example
    if not example:
        raise ValueError("assert_final_state_node requires dataset_example in state")

    # Initialize tools and tool entries
    tool_service = get_tool_service()
    await tool_service.initialize(state.context.mcp_services or [])
    tools_dict = await load_mcp_tools(state.context.mcp_services or [])

    # Prepare prompt context
    context_vars = extract_all_context_variables(
        user_context=state.context.user_context,
        github_context=state.context.github_context,
        mcp_resources=state.mcp_resources,
    )
    formatted_context_vars = format_context_variables_for_llm(context_vars)
    resource_hints = format_resource_hints(state.mcp_resources)

    instructions: Optional[List[str]] = None
    if example and example.expected_output:
        instructions = example.expected_output.assert_final_state or []

    if not instructions:
        raise ValueError(
            f"DatasetExample {example.example_id} has no assert_final_state instructions."
        )

    llm = ChatOpenAI(
        model="gpt-5",
        use_responses_api=True,
        output_version="responses/v1",
    )
    selected_tools = state.tool_selection_log.selected_tools

    actual_tools = [tools_dict[canonicalize_tool_name(tool)] for tool in selected_tools] + [web_search]

    assertion_agent = create_agent(
        model=llm,
        tools=actual_tools,
        system_prompt=SYSTEM_PROMPT,
        # response_format=FailureAnalysis,
        middleware=[handle_tool_errors]
    )

    user_prompt = HumanMessage(content=USER_PROMPT.format(criterias=instructions, resources=resource_hints, context=formatted_context_vars))

    result = await assertion_agent.ainvoke(input={"messages": [user_prompt]})
    completed_at = datetime.utcnow()
    assertion_output = await convert_response_v1_output_to_message_string(result)
    

    return {
        "assertion_output": assertion_output,
        "completed_at": completed_at,
    }


