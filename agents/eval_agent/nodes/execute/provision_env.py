"""Provision environment for a single dataset example based on natural language instructions."""
from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage
from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.resource_utils import format_resource_hints
from shared.llm import get_llm
from agents.eval_agent.reflexion_factory import create_ephemeral_reflexion
from langchain_core.runnables import RunnableConfig
from .utils import get_tool_hub

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
    instructions: List[str] =  example.expected_output.create_test_data

    # If no MCP services or no instructions, skip
    if not state.context.mcp_services or not instructions:
        logger.error("No provisioning instructions or MCP services; skipping provisioning phase.")
        return {}
    
    resource_hints = format_resource_hints(state.mcp_resources)

    tool_hub = await get_tool_hub(state)
    
    # Semantic Tool Selection: Use filtered tools if available
    tools_subset = None
    if state.tool_entries:
        # Resolve tool objects from the hub using the names selected in the plan phase
        tools_subset = [tool_hub.get_tool(name) for name in state.tool_entries.keys()]
        tools_subset = [t for t in tools_subset if t is not None]
        logger.info("Using subset of %d tools for provisioning", len(tools_subset))

    # Use Reflexion agent for provisioning
    provisioning_agent = create_ephemeral_reflexion(
        model=get_llm(model='gpt-4.1', temperature=0.0),
        tool_hub=tool_hub,
        tools=tools_subset, # Pass subset if available
        prompt=SYSTEM_PROMPT,
        agent_id="eval_provisioner_v1",
        max_rounds=2  # Limit rounds for provisioning to avoid infinite loops
    )
    user_prompt = HumanMessage(content=USER_PROMPT.format(instructions=instructions, resources=resource_hints))

    # Invoke with initial state
    result = await provisioning_agent.ainvoke(
        {"messages": [user_prompt], "current_round": 0}, 
        config=RunnableConfig(recursion_limit=75)
    )

    # Extract output from Reflexion state
    provisioning_output = result.get('candidate_response')
    if not provisioning_output and result.get('messages'):
        # Fallback to last message content if candidate_response is empty
        provisioning_output = result['messages'][-1].content

    return {
        "mcp_resources": state.mcp_resources,
        "started_at": started_at,
        "provisioning_output": provisioning_output,
    }
