import asyncio
from typing import Dict
from agents.eval_agent.models import EvalAgentPlannerState
from agents.eval_agent.nodes.execute.utils import get_tool_hub
from shared.logger import get_logger
from shared.tools import ToolEntry
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI
from pydantic import Field
from shared.mcp_client import ComposioMCPClient
from shared.config import COMPOSIO_USER_ID, config

logger = get_logger("eval_agent.plan.filter_tools")


class SelectedTool(BaseModel):
    selected_tools: List[str] = Field(description="list of tools that are required to complete the instructions")
    simulation_steps: List[str] = Field(description="list of steps to complete the instructions")


async def filter_tools(state: EvalAgentPlannerState) -> dict:
    """
    Filter tools using semantic search against the generated test plan.
    This reduces the context window and noise for the execution agents.
    """
    
    # TODO: cache all the tools so that we don't need to fetch them every time
    tool_service = ComposioMCPClient(["GITHUB", "ASANA"], COMPOSIO_USER_ID)
    all_tools = await tool_service.get_tools()
    llm = ChatOpenAI(model="gpt-5.1", reasoning_effort="medium")
    st = llm.with_structured_output(SelectedTool)
    tool_names = [{t.name,t.description} for t in all_tools]
    prompt = """
    basen on the instructions filter the tools that are required to complete the instructions.

    # Instructions
    {create_test_data}

    {assert_final_state}

    # Available tools
    {all_tools}

    # Important:
    - You should do a mental simlutaion to judge all tools that are required to complete the instructions.
    - you should include all the tools that are required to complete the instructions.
    - wtite down the simulation steps to complete the instructions.
    - always include some extra tools that may come handy, in case there isan issue
    - MUST include extra tools that can be used to fullfill the instructions in a different way than you have planned. There can always be a different way to fullfill the instructions.

    # Steps you should follow:
    1. Read the instructions carefully and understand the goal.
    2. Write down the simulation steps to complete the instructions one by one along with the tools that are required to complete those steps.
    3. think of other ways to fullfill the request that you have planned , for each step think of alternate ways to fullfill the request, with different tools.
    """
    create_test_data = state.dataset_examples[0].expected_output.create_test_data
    assert_final_state = state.dataset_examples[0].expected_output.assert_final_state

    input_message = prompt.format(create_test_data=create_test_data, assert_final_state=assert_final_state, all_tools=tool_names)
    output = await st.ainvoke(input=input_message)
    selected_tools = output.selected_tools

    tool_entries: Dict[str, ToolEntry] = {}
    for tool in all_tools:
        if tool.name in selected_tools:
            service = tool.name.split("_")[0]
            tool_entries[tool.name] = ToolEntry(
                name=tool.name,
                description=tool.description,
                service=service,
            )

    logger.info(f"Selected {len(tool_entries)} tools: {list(tool_entries.keys())}")

    return {
        "tool_entries": tool_entries,
    }
