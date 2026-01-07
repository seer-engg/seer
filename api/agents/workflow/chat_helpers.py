"""
Helper functions for workflow chat endpoint.

Extracted from router.py to reduce complexity of chat_with_workflow_endpoint.
"""
from typing import Optional, Dict, List, Any, Tuple, Set
from copy import deepcopy
import asyncio
import uuid

from fastapi import HTTPException
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from shared.logger import get_logger
from shared.config import config
from shared.analytics import analytics
from shared.database.models import User
from .services import (
    create_chat_session,
    get_chat_session,
    get_chat_session_by_thread_id,
    save_chat_message,
)
from .chat_schema import ChatResponse
from .models import WorkflowProposalPublic
from agents.workflow_agent import (
    set_workflow_state_for_thread,
    set_user_for_thread,
    _current_thread_id,
)
from api.agents.checkpointer import (
    get_checkpointer_with_retry,
    _recreate_checkpointer,
    get_checkpointer,
)

# Import psycopg for error type checking
try:
    import psycopg
except ImportError:
    psycopg = None

logger = get_logger(__name__)


# ===== Phase 1: Session Management =====


async def _get_or_create_session(thread_id: Optional[str], session_id: Optional[str], workflow, user: User, workflow_id: str) -> Tuple[Any, str, str]:
    """Get existing session or create new one. Returns (session, session_id, thread_id)."""
    session = None

    if thread_id:
        # Try to find existing session by thread_id
        session = await get_chat_session_by_thread_id(thread_id, workflow)
        if session:
            session_id = session.id
    elif session_id:
        # Get session by ID
        session = await get_chat_session(session_id, workflow)
        thread_id = session.thread_id

    # Create new session if needed
    if session is None:
        thread_id = thread_id or f"workflow-{workflow_id}-{uuid.uuid4().hex}"
        session = await create_chat_session(
            workflow=workflow,
            user=user,
            thread_id=thread_id,
        )
        session_id = session.id

    return session, session_id, thread_id


# ===== Phase 2: Workflow State Handling =====


def _prepare_workflow_state(workflow_state_snapshot: Dict[str, Any], provided_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge workflow state from database with frontend-provided changes."""
    # Deep copy to avoid mutations
    workflow_state = deepcopy(workflow_state_snapshot)

    if provided_state:
        # Merge provided nodes and edges
        provided_nodes = provided_state.get("nodes", [])
        provided_edges = provided_state.get("edges", [])
        workflow_state["nodes"] = provided_nodes if provided_nodes else workflow_state.get("nodes", [])
        workflow_state["edges"] = provided_edges if provided_edges else workflow_state.get("edges", [])

        # Merge any other keys
        for key, value in provided_state.items():
            if key not in ["nodes", "edges"]:
                workflow_state[key] = value

    # Ensure required keys exist
    if "nodes" not in workflow_state:
        workflow_state["nodes"] = []
    if "edges" not in workflow_state:
        workflow_state["edges"] = []

    return workflow_state


def _setup_thread_context(thread_id: str, workflow_state: Dict[str, Any], user: User) -> None:
    """Set workflow state and user in thread context for tools to access."""
    set_workflow_state_for_thread(thread_id, workflow_state)
    set_user_for_thread(thread_id, user)


# ===== Phase 3: Tool Call Detection =====


def _extract_tool_call_ids(msg: Any) -> Set[str]:
    """Extract tool call IDs from AIMessage."""
    tool_call_ids = set()

    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            # Handle both dict and object formats
            if isinstance(tc, dict):
                tool_call_id = tc.get("id")
            else:
                tool_call_id = getattr(tc, "id", None)
            if tool_call_id:
                tool_call_ids.add(tool_call_id)

    return tool_call_ids


def _extract_tool_response_ids(messages: List[Any]) -> Set[str]:
    """Extract tool call IDs from ToolMessages."""
    tool_response_ids = set()

    for m in messages:
        if isinstance(m, ToolMessage):
            tool_call_id = getattr(m, "tool_call_id", None)
            if tool_call_id:
                tool_response_ids.add(tool_call_id)
        elif isinstance(m, dict) and m.get("type") == "tool":
            tool_call_id = m.get("tool_call_id")
            if tool_call_id:
                tool_response_ids.add(tool_call_id)

    return tool_response_ids


def _has_incomplete_tool_calls(messages: List[Any]) -> bool:
    """
    Check if messages contain incomplete tool calls.

    Returns True if any AIMessage with tool_calls is missing corresponding ToolMessages.
    """
    for i, msg in enumerate(messages):
        # Check if this is an AIMessage with tool_calls
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_ids = _extract_tool_call_ids(msg)
            if not tool_call_ids:
                continue

            # Check following messages for ToolMessages
            following_msgs = messages[i+1:i+1+len(tool_call_ids)*2]
            tool_response_ids = _extract_tool_response_ids(following_msgs)

            # If any tool_call_ids don't have responses, it's incomplete
            if tool_call_ids - tool_response_ids:
                logger.warning(
                    f"Found incomplete tool calls. Missing responses for: {tool_call_ids - tool_response_ids}"
                )
                return True

    return False


async def _check_checkpoint_for_incomplete_tools(
    agent,
    config_dict: Dict[str, Any],
    thread_id: str,
) -> bool:
    """
    Check if checkpoint contains incomplete tool calls.

    Handles connection errors and retries.
    """
    logger.debug(f"Checking checkpointer health for thread {thread_id}")

    try:
        # Check health and reconnect if needed
        checkpointer = await get_checkpointer_with_retry()
        if checkpointer is None:
            logger.warning("Checkpointer unavailable, proceeding without state check")
            return False

        current_state = await agent.aget_state(config_dict)
        messages = current_state.values.get("messages", [])
        return _has_incomplete_tool_calls(messages)

    except (Exception, ConnectionError, EOFError) as e:
        # Check if it's a connection error
        is_connection_error = (
            (psycopg and isinstance(e, psycopg.OperationalError)) or
            isinstance(e, ConnectionError) or
            isinstance(e, EOFError) or
            "connection is closed" in str(e).lower() or
            "ssl syscall error" in str(e).lower()
        )

        if is_connection_error:
            logger.warning(f"Connection error during state check: {e}, attempting reconnection...")
            try:
                checkpointer = await _recreate_checkpointer()
                if checkpointer:
                    # Retry once after reconnection
                    current_state = await agent.aget_state(config_dict)
                    messages = current_state.values.get("messages", [])
                    return _has_incomplete_tool_calls(messages)
                else:
                    logger.warning("Failed to recreate checkpointer, proceeding without state check")
                    return False
            except Exception as reconnect_error:
                logger.error(f"Error during checkpointer reconnection: {reconnect_error}")
                return False
        else:
            logger.warning(f"Error checking state for incomplete tool calls: {e}. Proceeding with normal invocation.")
            return False


# ===== Phase 4: Checkpoint Recovery =====


async def _find_safe_checkpoint(checkpointer, config_dict: Dict[str, Any], thread_id: str):
    """
    Find the last checkpoint that doesn't have incomplete tool calls.

    Returns safe checkpoint tuple or None.
    """
    try:
        # List checkpoints with timeout
        async def list_checkpoints():
            return [c async for c in checkpointer.alist(config_dict)]

        checkpoints = await asyncio.wait_for(
            list_checkpoints(),
            timeout=10.0
        )
    except asyncio.TimeoutError:
        logger.error(f"Checkpoint listing timed out for thread {thread_id}")
        return None

    # Find the last checkpoint without incomplete tool calls
    for checkpoint_tuple in reversed(checkpoints[:-1]):  # Skip the latest (incomplete) one
        checkpoint_messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])

        # Check if this checkpoint has incomplete tool calls
        if not _has_incomplete_tool_calls(checkpoint_messages):
            return checkpoint_tuple

    return None


async def _delete_thread_with_retry(checkpointer, thread_id: str) -> None:
    """Delete thread with connection retry logic."""
    if hasattr(checkpointer, 'adelete_thread'):
        await checkpointer.adelete_thread(thread_id)
    else:
        await asyncio.to_thread(checkpointer.delete_thread, thread_id)


def _is_connection_error(e: Exception) -> bool:
    """Check if exception is a connection error."""
    return (
        (psycopg and isinstance(e, psycopg.OperationalError)) or
        isinstance(e, ConnectionError) or
        isinstance(e, EOFError) or
        "connection is closed" in str(e).lower() or
        "ssl syscall error" in str(e).lower()
    )


async def _invoke_fresh_agent(agent, user_msg: HumanMessage, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke agent with fresh start (no checkpoint resume)."""
    return await _invoke_agent_with_timeout(agent, {"messages": [user_msg]}, config_dict)


async def _invoke_agent_with_timeout(
    agent,
    messages: Dict[str, List[Any]],
    config: Dict[str, Any],
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """
    Invoke agent with timeout to prevent indefinite hangs.

    Sets thread_id in context variable for tools to access.
    """
    thread_id = config.get('configurable', {}).get('thread_id') if config else None

    # Set thread_id in context variable for tools
    if thread_id:
        token = _current_thread_id.set(thread_id)
    else:
        token = None

    try:
        return await asyncio.wait_for(
            agent.ainvoke(messages, config=config),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Agent invocation timed out after {timeout} seconds for thread {thread_id or 'unknown'}")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. The agent took too long to respond."
        )
    finally:
        # Reset context variable
        if token is not None:
            _current_thread_id.reset(token)


async def _resume_from_safe_checkpoint(
    agent, thread_id: str, safe_checkpoint, user_msg: HumanMessage
) -> Dict[str, Any]:
    """Resume agent from a safe checkpoint."""
    prev_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": safe_checkpoint.config["configurable"]["checkpoint_id"]
        }
    }
    logger.info(f"Resuming from safe checkpoint: {prev_config['configurable']['checkpoint_id']}")
    return await _invoke_agent_with_timeout(agent, {"messages": [user_msg]}, prev_config)


async def _handle_no_safe_checkpoint(
    checkpointer, thread_id: str, agent, user_msg: HumanMessage, config_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle case when no safe checkpoint found - delete thread and start fresh."""
    logger.warning(f"No safe checkpoint found, deleting thread {thread_id} and starting fresh")
    await _delete_thread_with_retry(checkpointer, thread_id)
    return await _invoke_fresh_agent(agent, user_msg, config_dict)


async def _cleanup_thread_fallback(thread_id: str) -> None:
    """Fallback cleanup: try to delete thread with fresh checkpointer."""
    try:
        fresh_checkpointer = await get_checkpointer()
        if fresh_checkpointer:
            await _delete_thread_with_retry(fresh_checkpointer, thread_id)
    except Exception:
        pass


async def _handle_recovery_connection_error(e: Exception, thread_id: str) -> None:
    """Handle connection errors during recovery by attempting reconnection."""
    logger.warning(f"Connection error during recovery: {e}, attempting reconnection...")
    try:
        checkpointer = await _recreate_checkpointer()
        if checkpointer:
            await _delete_thread_with_retry(checkpointer, thread_id)
    except Exception as reconnect_error:
        logger.error(f"Error during reconnection in recovery: {reconnect_error}")


async def _recover_from_incomplete_state(
    agent,
    checkpointer,
    thread_id: str,
    config_dict: Dict[str, Any],
    user_msg: HumanMessage,
) -> Dict[str, Any]:
    """
    Recover from incomplete tool call state.

    Tries to find a safe checkpoint, or deletes thread and starts fresh.
    """
    logger.warning(f"Incomplete tool calls detected in thread {thread_id}, attempting recovery...")

    if not checkpointer:
        logger.error("No checkpointer available for state recovery")
        return await _invoke_fresh_agent(agent, user_msg, config_dict)

    try:
        checkpointer = await get_checkpointer_with_retry()
        if checkpointer is None:
            logger.warning("Checkpointer unavailable for recovery, starting fresh")
            fresh_checkpointer = await get_checkpointer()
            if fresh_checkpointer:
                await _delete_thread_with_retry(fresh_checkpointer, thread_id)
            return await _invoke_fresh_agent(agent, user_msg, config_dict)

        # Try to find a safe checkpoint
        safe_checkpoint = await _find_safe_checkpoint(checkpointer, config_dict, thread_id)

        if safe_checkpoint:
            return await _resume_from_safe_checkpoint(agent, thread_id, safe_checkpoint, user_msg)
        else:
            return await _handle_no_safe_checkpoint(checkpointer, thread_id, agent, user_msg, config_dict)

    except (Exception, ConnectionError, EOFError) as e:
        if _is_connection_error(e):
            await _handle_recovery_connection_error(e, thread_id)
        else:
            logger.error(f"Error recovering from incomplete state: {e}", exc_info=True)

        await _cleanup_thread_fallback(thread_id)
        return await _invoke_fresh_agent(agent, user_msg, config_dict)


async def _invoke_with_checkpoint_recovery(
    agent,
    checkpointer,
    thread_id: str,
    config_dict: Dict[str, Any],
    user_msg: HumanMessage,
) -> Dict[str, Any]:
    """
    Invoke agent with automatic checkpoint recovery if incomplete tool calls detected.
    """
    # Check for incomplete tool calls before invoking
    has_incomplete = False
    if checkpointer and thread_id:
        has_incomplete = await _check_checkpoint_for_incomplete_tools(agent, config_dict, thread_id)

    if has_incomplete:
        # Recover from incomplete state
        return await _recover_from_incomplete_state(
            agent,
            checkpointer,
            thread_id,
            config_dict,
            user_msg,
        )
    else:
        # Normal invocation
        logger.info(f"Invoking agent for thread {thread_id} with checkpointer={'enabled' if checkpointer else 'disabled'}")
        result = await _invoke_agent_with_timeout(
            agent,
            {"messages": [user_msg]},
            config_dict,
        )
        logger.debug(f"Agent invocation completed for thread {thread_id}")
        return result


# ===== Phase 5: Interrupt Handling =====


def _extract_interrupt_data_from_value(interrupt_value: Any) -> Dict[str, Any]:
    """Extract interrupt data from various interrupt value formats."""
    if hasattr(interrupt_value, 'value'):
        value = interrupt_value.value
        return value if isinstance(value, dict) else {"value": value}
    elif isinstance(interrupt_value, dict):
        return interrupt_value.get('value', interrupt_value)
    else:
        return {"value": str(interrupt_value)}


def _check_result_for_interrupts(result: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check result dict for interrupt signals. Returns (interrupt_found, interrupt_data)."""
    if not isinstance(result, dict):
        return False, None

    if "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        if isinstance(interrupts, list) and len(interrupts) > 0:
            return True, _extract_interrupt_data_from_value(interrupts[0])
        elif isinstance(interrupts, dict):
            return True, interrupts
        else:
            return True, {"value": str(interrupts)}
    elif "interrupt" in result:
        interrupt_value = result["interrupt"]
        interrupt_data = interrupt_value if isinstance(interrupt_value, dict) else {"value": interrupt_value}
        return True, interrupt_data

    return False, None


async def _check_state_for_interrupts(agent, config_dict: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check agent state for interrupt signals. Returns (interrupt_found, interrupt_data)."""
    try:
        current_state = await agent.aget_state(config_dict)
        if not (hasattr(current_state, "interrupt") and current_state.interrupt):
            return False, None

        interrupt = current_state.interrupt
        if isinstance(interrupt, list) and len(interrupt) > 0:
            return True, _extract_interrupt_data_from_value(interrupt[0])
        elif isinstance(interrupt, dict):
            return True, interrupt
        else:
            return True, {"value": interrupt}
    except Exception as e:
        logger.debug(f"Could not check state for interrupts: {e}")
        return False, None


async def _check_for_interrupts(
    result: Dict[str, Any],
    agent,
    config_dict: Dict[str, Any],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check for interrupts in both result and current state.

    Returns:
        Tuple of (interrupt_required, interrupt_data)
    """
    # Check result for interrupts
    interrupt_required, interrupt_data = _check_result_for_interrupts(result)

    # Check current state for interrupts if not found in result
    if not interrupt_required:
        interrupt_required, interrupt_data = await _check_state_for_interrupts(agent, config_dict)

    return interrupt_required, interrupt_data


# ===== Phase 6: Response Processing =====


def _extract_response_text(result: Dict[str, Any]) -> str:
    """Extract response text from agent result."""
    agent_messages = result.get("messages", []) if isinstance(result, dict) else []

    if not agent_messages:
        return "I'm here to help with your workflow!"

    # Get last assistant message
    last_msg = agent_messages[-1]
    if hasattr(last_msg, "content"):
        return last_msg.content
    else:
        return str(last_msg)


async def _verify_checkpoint_saved(checkpointer, thread_id: str) -> None:
    """Verify that checkpoint was saved after agent invocation."""
    if not checkpointer or not thread_id:
        return

    try:
        verify_config = {"configurable": {"thread_id": thread_id}}
        state_tuple = await checkpointer.aget_tuple(verify_config)

        if state_tuple:
            checkpoint_id = state_tuple.config.get("configurable", {}).get("checkpoint_id")
            logger.info(f"Checkpoint verified for thread {thread_id}, checkpoint_id={checkpoint_id}")
        else:
            logger.warning(f"No checkpoint found for thread {thread_id} after agent invocation")
    except Exception as e:
        logger.error(f"Error verifying checkpoint for thread {thread_id}: {e}", exc_info=True)


async def _save_assistant_message_to_db(
    session_id: str,
    response_text: str,
    thinking_steps: Optional[List[str]],
    proposal_payload: Optional[Dict[str, Any]],
    proposal: Optional[Any],
) -> None:
    """Save assistant message to database."""
    await save_chat_message(
        session_id=session_id,
        role="assistant",
        content=response_text,
        thinking="\n".join(thinking_steps) if thinking_steps else None,
        suggested_edits=proposal_payload,
        proposal=proposal,
    )


def _track_assistant_message_analytics(
    user: User,
    workflow_id: str,
    session_id: str,
    response_text: str,
    model: str,
    proposal_public: Optional[WorkflowProposalPublic],
) -> None:
    """Track assistant message analytics event."""
    analytics.capture(
        distinct_id=user.user_id,
        event="chat_agent_message",
        properties={
            "workflow_id": workflow_id,
            "session_id": session_id,
            "message_role": "assistant",
            "message_length": len(response_text),
            "model": model,
            "created_proposal": proposal_public is not None,
            "deployment_mode": config.seer_mode,
        },
    )


def _build_chat_response(
    response_text: str,
    proposal_public: Optional[WorkflowProposalPublic],
    proposal_error: Optional[str],
    session_id: str,
    thread_id: str,
    thinking_steps: Optional[List[str]],
    interrupt_required: bool,
    interrupt_data: Optional[Dict[str, Any]],
) -> ChatResponse:
    """Build final chat response object."""
    return ChatResponse(
        response=response_text,
        proposal=proposal_public,
        proposal_error=proposal_error,
        session_id=session_id,
        thread_id=thread_id,
        thinking=thinking_steps if thinking_steps else None,
        interrupt_required=interrupt_required,
        interrupt_data=interrupt_data,
    )
