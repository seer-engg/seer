"""Invoke the target agent for a single dataset example."""
from datetime import datetime

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.test_runner.agent_invoker import invoke_target_agent

logger = get_logger("eval_agent.execute.invoke")


async def invoke_target_node(state: TestExecutionState) -> dict:
    """Invoke the target agent with the dataset example's input message."""
    example = state.dataset_example
    if not example:
        raise ValueError("invoke_target_node requires dataset_example in state")
    try:
        result = await invoke_target_agent(
            sandbox_context=state.context.sandbox_context,
            github_context=state.context.github_context,
            input_message=example.input_message,
            timeout_seconds=600,
        )

        return {
            "thread_id": result.thread_id,
            "agent_output": result.final_output or "",
        }
    except Exception as e:
        logger.error(f"Error invoking target agent: {e}")
        completed_at = datetime.utcnow()

        return {
            "assertion_output": str(e),
            "completed_at": completed_at,
        }


