"""Invoke the target agent for a single dataset example."""
from datetime import datetime

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.test_runner.agent_invoker import invoke_target_agent
from shared.message_formatter import format_target_agent_message
from shared.config import config
from langchain_core.messages import ToolMessage
import uuid
logger = get_logger("eval_agent.execute.invoke")


async def invoke_target_node(state: TestExecutionState) -> dict:
    """Invoke the target agent with the dataset example's input message."""
    example = state.dataset_example
    if not example:
        raise ValueError("invoke_target_node requires dataset_example in state")
    
    # Format message with appropriate context level
    context_level = config.target_agent_context_level
    
    formatted_message = format_target_agent_message(
        example=example,
        context=state.context,
        context_level=context_level
    )
    
    # Strategic logging: Log what we're sending to target agent
    logger.info(
        f"🎯 Invoking target agent: context_level={context_level}, "
        f"base_msg_len={len(example.input_message)}, formatted_msg_len={len(formatted_message)}, "
        f"msg_preview={formatted_message[:100]}..."
    )
    
    try:
        result = await invoke_target_agent(
            sandbox_context=state.context.sandbox_context,
            agent_name=state.context.agent_name,
            input_message=formatted_message,
            timeout_seconds=600,
        )

        output_messages = [ToolMessage(content=result.final_output or "", tool_call_id=str(uuid.uuid4()))]

        return {
            "thread_id": result.thread_id,
            "agent_output": result.final_output or "",
            "messages": output_messages,
        }
    except Exception as e:
        logger.error(f"Error invoking target agent: {e}")
        completed_at = datetime.utcnow()

        return {
            "assertion_output": str(e),
            "completed_at": completed_at,
        }


