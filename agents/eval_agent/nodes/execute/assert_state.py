"""Assert final state for a single dataset example based on natural language instructions."""
from datetime import datetime
from typing import List, Optional

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.resource_utils import format_resource_hints
from shared.parameter_population import (
    extract_all_context_variables,
    format_context_variables_for_llm,
)
from shared.llm import get_llm
from agents.eval_agent.reflexion_factory import create_ephemeral_reflexion
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from .utils import get_tool_hub


logger = get_logger("eval_agent.execute.assert")



SYSTEM_PROMPT = """
You are a helpful assistant that asserts the final state of the environment for the target agent. based on the specified criterias. you will use all the tools available to you to assert the final state.
"""
USER_PROMPT = """
Assert the following criterias by using the tools available to you:
{criterias}

previously anothe agent has done provisioning and it's output is:
<provisioning_output>
{provisioning_output}
</provisioning_output>
"""


async def assert_final_state_node(state: TestExecutionState) -> dict:
    """Plan and execute assertion steps, then produce ExperimentResultContext."""
    example = state.dataset_example
    if not example:
        raise ValueError("assert_final_state_node requires dataset_example in state")
    provisioning_output = state.provisioning_output

    # Initialize tools and tool entries

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

    tool_hub = await get_tool_hub(state)
    
    # Semantic Tool Selection: Use filtered tools if available
    tools_subset = None
    if state.tool_entries:
        # Resolve tool objects from the hub using the names selected in the plan phase
        tools_subset = [tool_hub.get_tool(name) for name in state.tool_entries.keys()]
        tools_subset = [t for t in tools_subset if t is not None]
        logger.info("Using subset of %d tools for assertion", len(tools_subset))

    assertion_agent = create_ephemeral_reflexion(
        model=get_llm(model='gpt-4.1', temperature=0.0),
        tool_hub=tool_hub,
        tools=tools_subset, # Pass subset if available
        prompt=SYSTEM_PROMPT,
        agent_id="eval_asserter_v1",
        max_rounds=2 
    )

    user_prompt = HumanMessage(content=USER_PROMPT.format(criterias=instructions, provisioning_output=provisioning_output))

    result = await assertion_agent.ainvoke(
        {
            "messages": [user_prompt], 
            "current_round": 0,
            "rubric": str(instructions) # Pass the rubric to the state
        },
        config=RunnableConfig(recursion_limit=75)
    )
    completed_at = datetime.utcnow()
    
    assertion_output = result.get('candidate_response')
    if not assertion_output and result.get('messages'):
        assertion_output = result['messages'][-1].content
    

    return {
        "assertion_output": assertion_output,
        "completed_at": completed_at,
    }
