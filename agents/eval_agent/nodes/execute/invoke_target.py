"""Invoke the target agent for a single dataset example."""
from datetime import datetime

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.test_runner.agent_invoker import invoke_target_agent
from shared.mcp_client import get_mcp_client_and_configs

logger = get_logger("eval_agent.execute.invoke")


async def invoke_target_node(state: TestExecutionState) -> dict:
    """Invoke the target agent with the dataset example's input message."""
    example = state.dataset_example
    if not example:
        raise ValueError("invoke_target_node requires dataset_example in state")

    # Prepare MCP configs to pass to the agent (if any)
    _, mcp_configs = await get_mcp_client_and_configs(state.context.mcp_services or [])

    result = await invoke_target_agent(
        sandbox_context=state.context.sandbox_context,
        github_context=state.context.github_context,
        input_message=example.input_message,
        mcp_resources=state.mcp_resources or {},
        mcp_configs=mcp_configs,
        timeout_seconds=300,
    )

    return {
        "thread_id": result.thread_id,
        "agent_output": result.final_output or "",
    }


