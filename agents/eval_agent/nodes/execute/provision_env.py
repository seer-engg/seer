"""Provision environment for a single dataset example based on natural language instructions."""
from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage
from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.llm import get_llm
from agents.eval_agent.reflexion_factory import create_ephemeral_reflexion
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from shared.tools import ComposioMCPClient
from shared.config import config
from .utils import handle_tool_errors
logger = get_logger("eval_agent.execute.provision")


from shared.prompt_loader import load_prompt

# Load prompts from YAML
_PROMPT_CONFIG = load_prompt("eval_agent/provision.yaml")
SYSTEM_PROMPT = _PROMPT_CONFIG.system
USER_PROMPT = _PROMPT_CONFIG.user_template




async def provision_environment_node(state: TestExecutionState) -> dict:
    """Plan and execute provisioning steps for the test example."""
    # Set start time if not set
    started_at = datetime.utcnow()

    example = state.dataset_example
    instructions: List[str] =  []
    for service_instructions in example.expected_output.create_test_data:
        instructions.extend(service_instructions.instructions)

    # If no MCP services or no instructions, skip
    if not state.context.mcp_services or not instructions:
        logger.error("No provisioning instructions or MCP services; skipping provisioning phase.")
        return {}
    
    resource_hints = state.mcp_resources

    # tool_hub = await get_tool_hub()    
    # Semantic Tool Selection: Use filtered tools if available
    # tools_subset = None
    # if state.tool_entries:
    #     # Resolve tool objects from the hub using the names selected in the plan phase
    #     tools_subset = [tool_hub.get_tool(name) for name in state.tool_entries.keys()]
    #     tools_subset = [t for t in tools_subset if t is not None]
    #     logger.info("Using subset of %d tools for provisioning", len(tools_subset))

    # Use Reflexion agent for provisioning
    # provisioning_agent = create_ephemeral_reflexion(
    #     model=get_llm(model='gpt-5.1', temperature=0.0),
    #     tool_hub=tool_hub,
    #     tools=tools_subset, # Pass subset if available
    #     prompt=SYSTEM_PROMPT,
    #     agent_id="eval_provisioner_v1",
    #     max_rounds=2  # Limit rounds for provisioning to avoid infinite loops
    # )

    tool_service = ComposioMCPClient(["GITHUB", "ASANA"], config.composio_user_id)
    all_tools = await tool_service.get_tools()

    llm = get_llm(model='gpt-5.1', reasoning_effort='high')
    actual_tools = []
    for tool in all_tools:
        if tool.name in state.context.tool_entries.keys():
            actual_tools.append(tool)
 
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