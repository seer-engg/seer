"""
Background tasks for async chat execution.

Provides resilience to client disconnections by running chat agent
execution in background workers.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from tortoise.exceptions import DoesNotExist

from seer.agents.nexus import (
    _current_thread_id,
    create_nexus_chat_agent,
    extract_thinking_from_messages,
)
from seer.agents.nexus.cost_callback import (
    CostCapCallbackHandler,
    set_chat_runtime_context,
    clear_chat_runtime_context,
)
from seer.api.agents.checkpointer import _recreate_checkpointer, get_checkpointer
from seer.api.agents.workflow.chat_services import (
    ChatOrchestrator,
    CheckpointerHealthService,
    IncompleteToolCallDetector,
    IncompleteToolCallRecoveryService,
    InterruptHandler,
)
from seer.utilities.langfuse_tracing import merge_nexus_langfuse_callbacks, langfuse_user_context
from seer.api.agents.workflow.services import (
    save_chat_message,
)
from seer.config import config
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.database import User
from seer.database.models import UserSettings
from seer.database.workflow_models import (
    ChatExecutionStatus,
    Workflow,
    WorkflowChatSession,
    WorkflowProposal,
)
from seer.logger import get_logger
from seer.observability.exceptions import RunCostCapExceeded
from seer.observability.sentry_client import set_user_context, set_tag, set_context
from seer.worker.broker_instance import broker
from seer.worker.tasks.memory import extract_session_memories

logger = get_logger(__name__)


def _set_sentry_context_for_chat(
    user: User,
    session_id: int,
    workflow_id: int,
    thread_id: Optional[str] = None,
) -> None:
    """
    Set Sentry context for chat execution error tracking.

    Sets user context (id, email, username) and chat session context.
    All operations are non-blocking and fail silently.
    """
    set_tag("task_type", "chat_execution")
    set_tag("session_id", str(session_id))
    set_tag("workflow_id", str(workflow_id))

    try:
        set_user_context(
            user_id=user.user_id,
            email=getattr(user, "email", None),
            username=f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip() or None,
        )
        # Set user tags for indexed searching in Sentry
        set_tag("user_id", user.user_id)
        if getattr(user, "email", None):
            set_tag("user_email", user.email)

        set_context("chat_session", {
            "session_id": session_id,
            "workflow_id": workflow_id,
            "thread_id": thread_id,
        })
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: Sentry context setup must never block task execution
        logger.debug("Failed to set Sentry context for chat", exc_info=True)


def _extract_response_text(result: Dict[str, Any]) -> str:
    """Extract response text from agent result."""
    agent_messages = result.get("messages", []) if isinstance(result, dict) else []
    if agent_messages:
        last_msg = agent_messages[-1]
        return last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return "I'm here to help with your workflow!"


async def _get_user_settings_and_context(
    user: User,
    thread_id: str,
) -> tuple[int, WorkflowRuntimeContext]:
    """Get user settings and create runtime context for cost tracking."""
    try:
        user_settings = await UserSettings.get(user=user)
        max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
        per_run_cost_cap_usd = user_settings.preferences.get("per_run_cost_cap_usd", 5.0)
    except DoesNotExist:
        max_agent_steps = config.nexus_max_agent_steps
        per_run_cost_cap_usd = 5.0

    runtime_context = WorkflowRuntimeContext(
        user=user,
        workflow_run_id=None,
        thread_id=thread_id,
        per_run_cost_cap_usd=per_run_cost_cap_usd,
        accumulated_cost_usd=0.0,
    )

    return max_agent_steps, runtime_context


async def _handle_agent_result(
    session: "WorkflowChatSession",
    result: Dict[str, Any],
    thread_id: str,
    session_id: int,
) -> None:
    """Handle agent result: detect interrupts, save messages, update session status."""
    interrupt_required, interrupt_data = InterruptHandler.extract_interrupt_from_result(result)
    agent_messages = result.get("messages", []) if isinstance(result, dict) else []
    response_text = _extract_response_text(result)
    thinking_steps = extract_thinking_from_messages(agent_messages)

    if interrupt_required and interrupt_data:
        logger.info(
            "Chat execution interrupted for user input",
            extra={"session_id": session_id, "interrupt_type": interrupt_data.get("type")}
        )
        session.current_execution_status = ChatExecutionStatus.INTERRUPTED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.pending_interrupt_type = interrupt_data.get("type")
        session.pending_interrupt_data = interrupt_data
        await session.save(update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "pending_interrupt_type",
            "pending_interrupt_data",
        ])
        await save_chat_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            thinking="\n".join(thinking_steps) if thinking_steps else None,
        )
    else:
        proposal = await WorkflowProposal.get_or_none(
            thread_id=thread_id,
            status=WorkflowProposal.STATUS_PENDING
        ).prefetch_related('created_by', 'workflow', 'session')

        await save_chat_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            thinking="\n".join(thinking_steps) if thinking_steps else None,
            suggested_edits=proposal.spec if proposal else None,
            proposal=proposal,
        )

        session.current_execution_status = ChatExecutionStatus.COMPLETED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_error = None
        await session.save(update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_error",
        ])

        logger.info(
            "Chat execution completed successfully",
            extra={"session_id": session_id}
        )

        if config.memory_enabled and config.memory_extraction_enabled:
            try:
                await extract_session_memories.kiq(session_id)
            except Exception:  # pylint: disable=broad-exception-caught # Reason: Non-critical background task
                logger.warning("Failed to enqueue memory extraction", extra={"session_id": session_id})


async def _invoke_agent_with_orchestrator(
    agent,
    checkpointer,
    user_msg: HumanMessage,
    thread_id: str,
    max_agent_steps: int,
) -> Dict[str, Any]:
    """Invoke agent using orchestrator with health checks."""
    cost_callback = CostCapCallbackHandler()

    config_dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_agent_steps,
        "callbacks": [cost_callback],
    }

    orchestrator = ChatOrchestrator(
        agent=agent,
        checkpointer=checkpointer,
        health_service=CheckpointerHealthService(),
        detector=IncompleteToolCallDetector(),
        recovery_service=IncompleteToolCallRecoveryService(),
        reconnect_func=_recreate_checkpointer,
    )

    return await orchestrator.invoke_with_health_checks(user_msg, config_dict)


@broker.task
async def chat_execution_task(
    # pylint: disable=too-many-positional-arguments,too-many-locals
    # Reason: Background task requires all parameters; multiple variables needed for orchestration
    session_id: int,
    user_id: int,
    message: str,
    workflow_id: int,
    model: Optional[str] = None,
) -> None:
    """
    Execute chat agent asynchronously in background.

    Args:
        session_id: Chat session database ID
        user_id: User database ID
        message: User's chat message
        workflow_id: Workflow database ID (not public workflow_id)
        model: Optional model name override
    """
    logger.info(
        "Starting async chat execution",
        extra={
            "session_id": session_id,
            "user_id": user_id,
            "workflow_id": workflow_id,
        }
    )

    # Update session status to RUNNING
    session = await WorkflowChatSession.get(id=session_id)
    session.current_execution_status = ChatExecutionStatus.RUNNING
    session.current_execution_started_at = datetime.now(timezone.utc)
    await session.save(update_fields=[
        "current_execution_status",
        "current_execution_started_at",
    ])

    try:
        # Fetch related entities
        user = await User.get(id=user_id)
        workflow = await Workflow.get(id=workflow_id)
        checkpointer = await get_checkpointer()
        thread_id = session.thread_id

        # Set Sentry context for error tracking
        _set_sentry_context_for_chat(user, session_id, workflow_id, thread_id)

        # Create agent (with memory context injection if enabled)
        agent = await create_nexus_chat_agent(
            model=model or config.default_llm_model,
            checkpointer=checkpointer,
            user_id=user.user_id,
            current_query=message,
            workflow_id=workflow.workflow_id,  # Public workflow ID for pre-bound tools
        )

        # Get user settings and create runtime context
        max_agent_steps, runtime_context = await _get_user_settings_and_context(user, thread_id)

        # Set context for callback access
        set_chat_runtime_context(runtime_context)

        try:
            # Invoke agent with Langfuse user context for trace attribution
            user_msg = HumanMessage(content=message)
            with langfuse_user_context(user.user_id):
                result = await _invoke_agent_with_orchestrator(
                    agent, checkpointer, user_msg, thread_id, max_agent_steps
                )

            await _handle_agent_result(session, result, thread_id, session_id)

        except RunCostCapExceeded as e:
            logger.warning(
                "Chat cost cap exceeded",
                extra={
                    "session_id": session_id,
                    "accumulated_cost": e.accumulated_cost,
                    "cost_cap": e.cost_cap,
                }
            )
            session.current_execution_status = ChatExecutionStatus.FAILED
            session.current_execution_finished_at = datetime.now(timezone.utc)
            session.current_execution_error = {
                "type": "cost_cap_exceeded",
                "detail": str(e.to_dict()),
                "status": 402,
            }
            await session.save(update_fields=[
                "current_execution_status",
                "current_execution_finished_at",
                "current_execution_error",
            ])
        finally:
            clear_chat_runtime_context()

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Background task must catch all exceptions to avoid worker crash
        logger.error(
            "Chat execution task failed",
            exc_info=True,
            extra={"session_id": session_id, "error": str(e)}
        )

        session.current_execution_status = ChatExecutionStatus.FAILED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_error = {
            "type": "execution_error",
            "detail": str(e),
            "status": 500,
        }
        await session.save(update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_error",
        ])


@broker.task
async def chat_resume_task(
    # pylint: disable=too-many-positional-arguments,too-many-locals
    # Reason: Background task requires all parameters; multiple variables needed for orchestration
    session_id: int,
    user_id: int,
    thread_id: str,
    resume_command_data: Dict[str, Any],
    workflow_id: int,
) -> None:
    """
    Resume chat execution after interrupt.

    Args:
        session_id: Chat session database ID
        user_id: User database ID
        thread_id: LangGraph thread ID
        resume_command_data: Serialized Command data (resume value)
        workflow_id: Workflow database ID (not public workflow_id)
    """
    logger.info(
        "Starting async chat resume",
        extra={
            "session_id": session_id,
            "user_id": user_id,
            "thread_id": thread_id,
        }
    )

    # Clear interrupt state and set status to RUNNING
    session = await WorkflowChatSession.get(id=session_id)
    session.current_execution_status = ChatExecutionStatus.RUNNING
    session.current_execution_started_at = datetime.now(timezone.utc)
    session.pending_interrupt_type = None
    session.pending_interrupt_data = None
    await session.save(update_fields=[
        "current_execution_status",
        "current_execution_started_at",
        "pending_interrupt_type",
        "pending_interrupt_data",
    ])

    try:
        # Fetch related entities
        user = await User.get(id=user_id)
        workflow = await Workflow.get(id=workflow_id)
        checkpointer = await get_checkpointer()

        # Set Sentry context for error tracking
        _set_sentry_context_for_chat(user, session_id, workflow_id, thread_id)

        # Create agent (with memory context - no query for resume)
        agent = await create_nexus_chat_agent(
            model=config.default_llm_model,
            checkpointer=checkpointer,
            user_id=user.user_id,
            workflow_id=workflow.workflow_id,  # Public workflow ID for pre-bound tools
        )

        # Build resume command
        resume_command = Command(resume=resume_command_data)

        # Get user settings and create runtime context for cost tracking
        max_agent_steps, runtime_context = await _get_user_settings_and_context(user, thread_id)

        # Set context for callback access
        set_chat_runtime_context(runtime_context)

        # Resume agent execution with cost callback
        cost_callback = CostCapCallbackHandler()
        config_dict = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max_agent_steps,
            "callbacks": [cost_callback],
        }
        config_with_langfuse = merge_nexus_langfuse_callbacks(config_dict)

        # Set thread_id in context variable
        token = _current_thread_id.set(thread_id)
        try:
            # Wrap agent invocation with Langfuse user context for trace attribution
            with langfuse_user_context(user.user_id):
                result = await agent.ainvoke(resume_command, config=config_with_langfuse)

            await _handle_agent_result(session, result, thread_id, session_id)

        finally:
            _current_thread_id.reset(token)
            clear_chat_runtime_context()

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Background task must catch all exceptions to avoid worker crash
        logger.error(
            "Chat resume task failed",
            exc_info=True,
            extra={"session_id": session_id, "error": str(e)}
        )

        session.current_execution_status = ChatExecutionStatus.FAILED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_error = {
            "type": "execution_error",
            "detail": str(e),
            "status": 500,
        }
        await session.save(update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_error",
        ])


@broker.task
async def cleanup_stale_chat_executions() -> None:
    """
    Reset stale QUEUED/RUNNING sessions (>1 hour old) to FAILED.

    Prevents sessions from getting stuck indefinitely due to worker crashes
    or other infrastructure issues.
    """
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

    stale_sessions = await WorkflowChatSession.filter(
        current_execution_status__in=[
            ChatExecutionStatus.QUEUED,
            ChatExecutionStatus.RUNNING
        ],
        current_execution_started_at__lt=cutoff_time,
    ).all()

    for session in stale_sessions:
        session.current_execution_status = ChatExecutionStatus.FAILED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_error = {
            "type": "timeout",
            "detail": "Execution timeout - session was stale",
            "reason": "cleanup_task",
            "status": 500,
        }
        await session.save(update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_error",
        ])

    logger.info("Cleaned up %d stale chat executions", len(stale_sessions))


__all__ = ["chat_execution_task", "chat_resume_task", "cleanup_stale_chat_executions", "_handle_agent_result"]
