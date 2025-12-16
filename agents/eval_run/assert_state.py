"""Assert final state for a single dataset example based on natural language instructions."""
from datetime import datetime
from typing import List, Optional

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.llm import get_llm_without_responses_api
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from shared.tools import ComposioMCPClient
from shared.config import config
from .utils import get_tool_hub, handle_tool_errors
from langchain.agents import create_agent
import uuid

logger = get_logger("eval_agent.execute.assert")



from shared.prompt_loader import load_prompt

# Load prompts from YAML
_PROMPT_CONFIG = load_prompt("eval_agent/assert.yaml")
SYSTEM_PROMPT = _PROMPT_CONFIG.system
USER_PROMPT = _PROMPT_CONFIG.user_template



async def assert_final_state_node(state: TestExecutionState) -> dict:
    """Plan and execute assertion steps, then produce ExperimentResultContext."""
    example = state.dataset_example
    if not example:
        raise ValueError("assert_final_state_node requires dataset_example in state")
    provisioning_output = state.provisioning_output

    instructions: Optional[List[str]] = []
    if example and example.expected_output:
        for service_instructions in example.expected_output.assert_final_state:
            instructions.extend(service_instructions.instructions)

    if not instructions:
        raise ValueError(
            f"DatasetExample {example.example_id} has no assert_final_state instructions."
        )

    llm = get_llm_without_responses_api()

    # TODO: replace hardcoded asana services
    tool_service = ComposioMCPClient(["GITHUB", "ASANA"], state.context.user_id)
    all_tools = await tool_service.get_tools()

    actual_tools = []
    for tool in all_tools:
        if tool.name in state.context.tool_entries.keys():
            actual_tools.append(tool)

    assertion_agent = create_agent(
        model=llm,
        tools=actual_tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[handle_tool_errors]
    )

    user_prompt = HumanMessage(content=USER_PROMPT.format(criterias=instructions, provisioning_output=provisioning_output))

    result = await assertion_agent.ainvoke(input={"messages": [user_prompt]})
    completed_at = datetime.utcnow()
    assertion_output = result.get('messages')[-1].content
    
    output_messages = [ToolMessage(content=assertion_output, tool_call_id=str(uuid.uuid4()))]

    return {
        "assertion_output": assertion_output,
        "completed_at": completed_at,
        "messages": output_messages,
    }
