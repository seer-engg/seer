"""
Streaming SSE helpers for workflow chat endpoints.

Extracted from chat_services.py to reduce module size.
"""
import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from seer.logger import get_logger

logger = get_logger(__name__)


async def stream_agent_tokens(
    agent: Any,
    input_data: Any,
    config: Dict[str, Any],
    user_id: str,
) -> AsyncGenerator[str, None]:
    """Stream tokens from agent via astream_events, yielding token strings."""
    from seer.utilities.langfuse_tracing import langfuse_user_context  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    with langfuse_user_context(user_id):
        async for event in agent.astream_events(input_data, config=config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk:
                    content = chunk.content if hasattr(chunk, "content") else ""
                    if content:
                        yield content


async def extract_post_stream_state(agent: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract response, thinking, and interrupt data from agent state after streaming."""
    from seer.agents.nexus import extract_thinking_from_messages  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from .chat_services import InterruptHandler  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    result = await agent.aget_state(config)
    full_response = ""
    all_thinking: List[str] = []

    if hasattr(result, 'values'):
        agent_messages = result.values.get("messages", [])
        all_thinking = extract_thinking_from_messages(agent_messages)
        if agent_messages:
            last_msg = agent_messages[-1]
            full_response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    interrupt_required, interrupt_data = await InterruptHandler.extract_interrupt_from_state(agent, config)

    return {
        "full_response": full_response,
        "all_thinking": all_thinking,
        "interrupt_required": interrupt_required,
        "interrupt_data": interrupt_data,
    }


async def save_stream_result(  # pylint: disable=too-many-positional-arguments # Reason: All params are distinct required context
    session: Any,
    session_id: int,
    thread_id: str,
    full_response: str,
    all_thinking: List[str],
    interrupt_required: bool,
    interrupt_data: Optional[Dict[str, Any]],
) -> Optional[Any]:
    """Save session status and chat message after streaming completes. Returns proposal if found."""
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.database.workflow_models import ChatExecutionStatus, WorkflowProposal
    from seer.worker.tasks.memory import extract_session_memories
    from seer.api.agents.workflow.services import save_chat_message
    from seer.config import config

    thinking_text = "\n".join(all_thinking) if all_thinking else None
    proposal = None

    if interrupt_required and interrupt_data:
        session.current_execution_status = ChatExecutionStatus.INTERRUPTED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.pending_interrupt_type = interrupt_data.get("type")
        session.pending_interrupt_data = interrupt_data
        await session.save(update_fields=[
            "current_execution_status", "current_execution_finished_at",
            "pending_interrupt_type", "pending_interrupt_data",
        ])
        await save_chat_message(session_id=session_id, role="assistant", content=full_response, thinking=thinking_text)
    else:
        proposal = await WorkflowProposal.get_or_none(thread_id=thread_id, status=WorkflowProposal.STATUS_PENDING)
        if proposal:
            await proposal.fetch_related('created_by', 'workflow', 'session')

        if not proposal:
            from seer.agents.nexus import _turn_completed_without_proposal  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
            if not _turn_completed_without_proposal.get():
                logger.warning(
                    "Agent completed without terminal tool (enforcement may have hit max retries)",
                    extra={"session_id": session_id, "thread_id": thread_id},
                )

        await save_chat_message(
            session_id=session_id, role="assistant", content=full_response, thinking=thinking_text,
            suggested_edits=proposal.spec if proposal else None, proposal=proposal,
        )
        session.current_execution_status = ChatExecutionStatus.COMPLETED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_error = None
        await session.save(update_fields=["current_execution_status", "current_execution_finished_at", "current_execution_error"])

        if config.memory_enabled and config.memory_extraction_enabled:
            try:
                await extract_session_memories.kiq(session_id)
            except Exception:  # pylint: disable=broad-exception-caught # Reason: Non-critical background task
                logger.warning("Failed to enqueue memory extraction", extra={"session_id": session_id})

    return proposal


def build_final_sse_event(  # pylint: disable=too-many-positional-arguments # Reason: All params needed for SSE event
    session_id: int,
    thread_id: str,
    all_thinking: List[str],
    interrupt_required: bool,
    interrupt_data: Optional[Dict[str, Any]],
    proposal: Optional[Any],
) -> str:
    """Build the final SSE done event JSON string."""
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.api.agents.workflow.router import _transform_clarification_interrupt
    from seer.api.agents.workflow.chat_schema import WorkflowProposalPublic

    final_event: Dict[str, Any] = {
        "done": True,
        "session_id": session_id,
        "thread_id": thread_id,
        "thinking": all_thinking or [],
        "interrupt_required": interrupt_required,
        "interrupt_data": None,
        "proposal": None,
    }
    if interrupt_data:
        _transform_clarification_interrupt(interrupt_data)
        final_event["interrupt_data"] = interrupt_data
    if not interrupt_required and proposal:
        final_event["proposal"] = WorkflowProposalPublic.model_validate(proposal, from_attributes=True).model_dump()

    return f"data: {json.dumps(final_event, default=str)}\n\n"


async def mark_session_failed(session: Any, error_type: str, detail: str, status: int = 500) -> None:
    """Mark a session as failed with error details."""
    from seer.database.workflow_models import ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    session.current_execution_status = ChatExecutionStatus.FAILED
    session.current_execution_finished_at = datetime.now(timezone.utc)
    session.current_execution_error = {"type": error_type, "detail": detail, "status": status}
    await session.save(update_fields=["current_execution_status", "current_execution_finished_at", "current_execution_error"])
