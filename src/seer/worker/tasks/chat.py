# pylint: disable=too-many-lines
# Reason: Background task module with complex agent orchestration; refactoring deferred
"""
Background tasks for async chat execution.

Provides resilience to client disconnections by running chat agent
execution in background workers.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from tortoise.exceptions import DoesNotExist

from seer.agents.nexus import (
    _current_thread_id,
    _submission_attempt_count,
    create_nexus_chat_agent,
    extract_thinking_from_messages,
)
from seer.agents.nexus.cost_callback import (
    CostCapCallbackHandler,
    set_chat_runtime_context,
    clear_chat_runtime_context,
)
from seer.agents.nexus.stream_publisher import StreamPublisher
from seer.api.agents.checkpointer import _recreate_checkpointer, get_checkpointer
from seer.api.agents.workflow.chat_schema import StreamEventType
from seer.api.agents.workflow.chat_services import (
    ChatOrchestrator,
    CheckpointerHealthService,
    IncompleteToolCallDetector,
    IncompleteToolCallRecoveryService,
    InterruptHandler,
    process_stream_event,
)
from seer.utilities.langfuse_tracing import (
    merge_nexus_langfuse_callbacks,
    langfuse_user_context,
)
from seer.api.agents.workflow.services import (
    save_chat_message,
)
from seer.config import config
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.services.workflows.mcp_config_adapter import McpServerConfigResolverImpl
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
from seer.worker.broker_instance import broker, create_redis_client
from seer.worker.tasks.memory import extract_session_memories

logger = get_logger(__name__)

try:
    import httpcore
except ImportError:
    httpcore = None

_TERMINAL_CHAT_STATUSES = {
    ChatExecutionStatus.COMPLETED,
    ChatExecutionStatus.FAILED,
    ChatExecutionStatus.INTERRUPTED,
}
_CHAT_LOCK_TTL_BUFFER_MS = 60000
_CHAT_LOCK_MIN_TTL_MS = 30000
_CHAT_LOCK_MAX_HEARTBEAT_INTERVAL_MS = 60000
_CHAT_LOCK_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""
_CHAT_LOCK_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""


@dataclass(slots=True)
class ChatExecutionContext:
    """Dependencies prepared before streaming a chat execution."""

    user: User
    checkpointer: Any
    thread_id: str
    agent: Any
    max_agent_steps: int
    runtime_context: WorkflowRuntimeContext


@dataclass(slots=True)
class ChatExecutionRequest:
    """Broker payload for a fresh chat execution."""

    session_id: int
    user_id: int
    message: str
    workflow_id: int
    execution_task_id: Optional[str]
    model: Optional[str]
    timezone: Optional[str] = None


@dataclass(slots=True)
class _ChatExecutionLockConfig:
    """Computed lock configuration for single-flight chat execution."""

    session_id: int
    execution_task_id: str
    task_name: str
    lock_key: str
    ttl_ms: int
    heartbeat_interval_seconds: float


class _ChatExecutionSingleFlightLock:
    """Redis-backed single-flight lock for chat execution ownership."""

    def __init__(
        self, session_id: int, execution_task_id: Optional[str], task_name: str
    ) -> None:
        resolved_execution_task_id = execution_task_id or "no-owner"
        ttl_ms = max(
            int(config.nexus_chat_timeout_seconds * 1000) + _CHAT_LOCK_TTL_BUFFER_MS,
            _CHAT_LOCK_MIN_TTL_MS,
        )
        self._config = _ChatExecutionLockConfig(
            session_id=session_id,
            execution_task_id=resolved_execution_task_id,
            task_name=task_name,
            lock_key=f"nexus:chat-execution:{session_id}:{resolved_execution_task_id}",
            ttl_ms=ttl_ms,
            heartbeat_interval_seconds=max(
                1.0,
                min(ttl_ms / 3000.0, _CHAT_LOCK_MAX_HEARTBEAT_INTERVAL_MS / 1000.0),
            ),
        )
        self._token = uuid4().hex
        self._redis = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None

    @property
    def lock_key(self) -> str:
        """Expose the Redis key for logging."""
        return self._config.lock_key

    async def acquire(self) -> bool:
        """Acquire the single-flight lock and start heartbeating it."""
        self._redis = create_redis_client(decode_responses=True)
        acquired = await self._redis.set(
            self._config.lock_key, self._token, nx=True, px=self._config.ttl_ms
        )
        if not acquired:
            await self._redis.aclose()
            self._redis = None
            return False

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat(),
            name=f"{self._config.task_name}-lock-heartbeat-{self._config.session_id}",
        )
        return True

    async def _heartbeat(self) -> None:
        """Renew the lock TTL while the worker is actively processing the chat."""
        try:
            while True:
                await asyncio.sleep(self._config.heartbeat_interval_seconds)
                renewed = await self._redis.eval(
                    _CHAT_LOCK_RENEW_SCRIPT,
                    1,
                    self._config.lock_key,
                    self._token,
                    str(self._config.ttl_ms),
                )
                if not renewed:
                    logger.warning(
                        "Chat execution lock heartbeat lost ownership",
                        extra={
                            "session_id": self._config.session_id,
                            "execution_task_id": self._config.execution_task_id,
                            "lock_key": self._config.lock_key,
                            "task_name": self._config.task_name,
                        },
                    )
                    return
        except Exception:  # pylint: disable=broad-exception-caught # Reason: Heartbeat failure should be logged but not crash the chat task
            logger.warning(
                "Failed to renew chat execution lock heartbeat",
                exc_info=True,
                extra={
                    "session_id": self._config.session_id,
                    "execution_task_id": self._config.execution_task_id,
                    "lock_key": self._config.lock_key,
                    "task_name": self._config.task_name,
                },
            )

    async def release(self) -> None:
        """Release the lock if this worker still owns it."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        if self._redis is None:
            return

        try:
            await self._redis.eval(
                _CHAT_LOCK_RELEASE_SCRIPT, 1, self._config.lock_key, self._token
            )
        finally:
            await self._redis.aclose()
            self._redis = None


class _NoOpChatExecutionSingleFlightLock:
    """No-op lock used when a chat task has no execution owner ID."""

    def __init__(self, session_id: int, task_name: str) -> None:
        self.session_id = session_id
        self.task_name = task_name
        self.lock_key = ""

    async def release(self) -> None:
        """No-op release for direct task invocations without owner metadata."""
        return


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
            username=f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip()
            or None,
        )
        # Set user tags for indexed searching in Sentry
        set_tag("user_id", user.user_id)
        if getattr(user, "email", None):
            set_tag("user_email", user.email)

        set_context(
            "chat_session",
            {
                "session_id": session_id,
                "workflow_id": workflow_id,
                "thread_id": thread_id,
            },
        )
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: Sentry context setup must never block task execution
        logger.debug("Failed to set Sentry context for chat", exc_info=True)


def _extract_response_text(result: Dict[str, Any]) -> str:
    """Extract response text from agent result."""
    agent_messages = result.get("messages", []) if isinstance(result, dict) else []
    if agent_messages:
        last_msg = agent_messages[-1]
        return last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return "I'm here to help with your workflow!"


async def _get_session_if_current_owner(
    session_id: int,
    execution_task_id: Optional[str],
    phase: str,
) -> Optional["WorkflowChatSession"]:
    """Return session only if this task still owns the current execution."""
    session = await WorkflowChatSession.get_or_none(id=session_id)
    if session is None:
        logger.info(
            "Skipping chat task during %s because session no longer exists",
            phase,
            extra={
                "session_id": session_id,
                "phase": phase,
                "task_execution_owner": execution_task_id,
            },
        )
        return None

    if execution_task_id and session.current_execution_task_id != execution_task_id:
        logger.info(
            "Skipping stale chat task during %s",
            phase,
            extra={
                "session_id": session_id,
                "phase": phase,
                "task_execution_owner": execution_task_id,
                "current_execution_task_id": session.current_execution_task_id,
            },
        )
        return None

    return session


async def _persist_failure_if_current_owner(
    session_id: int,
    execution_task_id: Optional[str],
    error_payload: Dict[str, Any],
) -> None:
    """Persist FAILED only if this task still owns the session."""
    session = await _get_session_if_current_owner(
        session_id, execution_task_id, "persist_failure"
    )
    if session is None:
        return

    if session.current_execution_status in _TERMINAL_CHAT_STATUSES:
        logger.info(
            "Skipping failure persistence for terminal chat session",
            extra={
                "session_id": session_id,
                "execution_task_id": execution_task_id,
                "current_execution_status": session.current_execution_status,
            },
        )
        return

    session.current_execution_status = ChatExecutionStatus.FAILED
    session.current_execution_finished_at = datetime.now(timezone.utc)
    session.current_execution_error = error_payload
    session.current_execution_task_id = None
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_error",
            "current_execution_task_id",
        ]
    )


def _iter_exception_chain(error: BaseException) -> list[BaseException]:
    """Return the exception, its cause chain, and any linked context exceptions."""
    exceptions: list[BaseException] = []
    seen: set[int] = set()
    current: Optional[BaseException] = error

    while current is not None and id(current) not in seen:
        exceptions.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return exceptions


def _is_upstream_stream_error(error: BaseException) -> bool:
    """Detect transport-level failures from the upstream LLM streaming stack."""
    for exc in _iter_exception_chain(error):
        if isinstance(exc, httpx.HTTPError):
            return True
        if httpcore is not None and isinstance(exc, httpcore.NetworkError):
            return True

        module_name = exc.__class__.__module__
        class_name = exc.__class__.__name__
        if module_name.startswith("httpcore") or module_name.startswith("httpx"):
            return True
        if module_name.startswith("openai") and class_name in {
            "APIConnectionError",
            "APITimeoutError",
        }:
            return True

    return False


def _build_execution_error_payload(error: BaseException) -> Dict[str, Any]:
    """Normalize exceptions into a persisted error payload."""
    raw_detail = str(error).strip()
    exception_type = f"{error.__class__.__module__}.{error.__class__.__name__}"

    if _is_upstream_stream_error(error):
        payload: Dict[str, Any] = {
            "type": "upstream_stream_error",
            "detail": "Upstream model stream disconnected before completion.",
            "status": 502,
            "exception_type": exception_type,
        }
        if raw_detail:
            payload["exception_detail"] = raw_detail
        return payload

    return {
        "type": "execution_error",
        "detail": raw_detail or error.__class__.__name__,
        "status": 500,
        "exception_type": exception_type,
    }


def _build_cancelled_execution_error_payload(error: BaseException) -> Dict[str, Any]:
    """Persist a terminal state when a chat task is cancelled."""
    return {
        "type": "execution_cancelled",
        "detail": "Chat execution was cancelled before completion.",
        "status": 503,
        "exception_type": f"{error.__class__.__module__}.{error.__class__.__name__}",
    }


def _should_abort_chat_execution(
    session: WorkflowChatSession,
    session_id: int,
    execution_task_id: Optional[str],
) -> bool:
    """Return True when a fresh execution should not proceed."""
    if (
        session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        or session.pending_interrupt_data
    ):
        logger.info(
            "Aborting fresh chat execution because session is already interrupted",
            extra={"session_id": session_id, "execution_task_id": execution_task_id},
        )
        return True

    return False


async def _mark_chat_execution_running(session: WorkflowChatSession) -> None:
    """Persist RUNNING status for a session."""
    session.current_execution_status = ChatExecutionStatus.RUNNING
    session.current_execution_started_at = datetime.now(timezone.utc)
    session.current_execution_finished_at = None
    session.current_execution_error = None
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_started_at",
            "current_execution_finished_at",
            "current_execution_error",
        ]
    )


async def _acquire_execution_single_flight_lock(
    session_id: int,
    execution_task_id: Optional[str],
    task_name: str,
) -> Optional[_ChatExecutionSingleFlightLock | _NoOpChatExecutionSingleFlightLock]:
    """Acquire a Redis single-flight lock or return None when a duplicate is already running."""
    if not execution_task_id:
        logger.info(
            "Skipping chat single-flight lock because execution owner ID is missing",
            extra={
                "session_id": session_id,
                "task_name": task_name,
            },
        )
        return _NoOpChatExecutionSingleFlightLock(
            session_id=session_id, task_name=task_name
        )

    lock = _ChatExecutionSingleFlightLock(session_id, execution_task_id, task_name)
    if await lock.acquire():
        return lock

    logger.warning(
        "Skipping duplicate chat task redelivery",
        extra={
            "session_id": session_id,
            "execution_task_id": execution_task_id,
            "lock_key": lock.lock_key,
            "task_name": task_name,
        },
    )
    return None


async def _get_user_settings_and_context(
    user: User,
    thread_id: str,
) -> tuple[int, WorkflowRuntimeContext]:
    """Get user settings and create runtime context for cost tracking."""
    try:
        user_settings = await UserSettings.get(user=user)
        max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
        per_run_cost_cap_usd = user_settings.preferences.get(
            "per_run_cost_cap_usd", 5.0
        )
    except DoesNotExist:
        max_agent_steps = config.nexus_max_agent_steps
        per_run_cost_cap_usd = 5.0

    runtime_context = WorkflowRuntimeContext(
        user=user,
        workflow_run_id=None,
        thread_id=thread_id,
        per_run_cost_cap_usd=per_run_cost_cap_usd,
        accumulated_cost_usd=0.0,
        mcp_config_resolver=McpServerConfigResolverImpl(user=user),
    )

    return max_agent_steps, runtime_context


async def _handle_agent_result(
    session: "WorkflowChatSession",
    result: Dict[str, Any],
    thread_id: str,
    session_id: int,
) -> None:
    """Handle agent result: detect interrupts, save messages, update session status."""
    interrupt_required, interrupt_data = InterruptHandler.extract_interrupt_from_result(
        result
    )
    agent_messages = result.get("messages", []) if isinstance(result, dict) else []
    response_text = _extract_response_text(result)
    thinking_steps = extract_thinking_from_messages(agent_messages)

    if interrupt_required and interrupt_data:
        logger.info(
            "Chat execution interrupted for user input",
            extra={
                "session_id": session_id,
                "interrupt_type": interrupt_data.get("type"),
            },
        )
        session.current_execution_status = ChatExecutionStatus.INTERRUPTED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_task_id = None
        session.pending_interrupt_type = interrupt_data.get("type")
        session.pending_interrupt_data = interrupt_data
        await session.save(
            update_fields=[
                "current_execution_status",
                "current_execution_finished_at",
                "current_execution_task_id",
                "pending_interrupt_type",
                "pending_interrupt_data",
            ]
        )
        await save_chat_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            thinking="\n".join(thinking_steps) if thinking_steps else None,
        )
    else:
        proposal = await WorkflowProposal.get_or_none(
            thread_id=thread_id, status=WorkflowProposal.STATUS_PENDING
        ).prefetch_related("created_by", "workflow", "session")

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
        session.current_execution_task_id = None
        await session.save(
            update_fields=[
                "current_execution_status",
                "current_execution_finished_at",
                "current_execution_error",
                "current_execution_task_id",
            ]
        )

        logger.info(
            "Chat execution completed successfully", extra={"session_id": session_id}
        )

        if config.memory_enabled and config.memory_extraction_enabled:
            try:
                await extract_session_memories.kiq(session_id)
            except Exception:  # pylint: disable=broad-exception-caught # Reason: Non-critical background task
                logger.warning(
                    "Failed to enqueue memory extraction",
                    extra={"session_id": session_id},
                )


async def _prepare_chat_execution(
    request: ChatExecutionRequest,
) -> Optional[ChatExecutionContext]:
    """Fetch dependencies and create the chat agent for execution."""
    user = await User.get(id=request.user_id)
    workflow = await Workflow.get(id=request.workflow_id)
    checkpointer = await get_checkpointer()
    session = await _get_session_if_current_owner(
        request.session_id,
        request.execution_task_id,
        "before_agent_create",
    )
    if session is None:
        return None

    thread_id = session.thread_id
    _set_sentry_context_for_chat(
        user, request.session_id, request.workflow_id, thread_id
    )

    agent = await create_nexus_chat_agent(
        model=request.model or config.default_llm_model,
        checkpointer=checkpointer,
        user_id=user.user_id,
        current_query=request.message,
        workflow_id=workflow.workflow_id,  # Public workflow ID for pre-bound tools
        timezone=request.timezone,
        organization_id=user.active_organization_id,
    )
    max_agent_steps, runtime_context = await _get_user_settings_and_context(
        user, thread_id
    )

    return ChatExecutionContext(
        user=user,
        checkpointer=checkpointer,
        thread_id=thread_id,
        agent=agent,
        max_agent_steps=max_agent_steps,
        runtime_context=runtime_context,
    )


async def _mark_chat_resume_running(session: WorkflowChatSession) -> None:
    """Persist RUNNING status for a resumed execution and clear stale interrupt state."""
    session.current_execution_status = ChatExecutionStatus.RUNNING
    session.current_execution_started_at = datetime.now(timezone.utc)
    session.current_execution_finished_at = None
    session.current_execution_error = None
    session.pending_interrupt_type = None
    session.pending_interrupt_data = None
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_started_at",
            "current_execution_finished_at",
            "current_execution_error",
            "pending_interrupt_type",
            "pending_interrupt_data",
        ]
    )


async def _prepare_chat_resume_execution(
    session_id: int,
    user_id: int,
    thread_id: str,
    workflow_id: int,
    execution_task_id: Optional[str],
) -> Optional[ChatExecutionContext]:
    """Fetch dependencies and create the chat agent for a resumed execution."""
    user = await User.get(id=user_id)
    workflow = await Workflow.get(id=workflow_id)
    checkpointer = await get_checkpointer()
    session = await _get_session_if_current_owner(
        session_id, execution_task_id, "before_resume_agent_create"
    )
    if session is None:
        return None

    _set_sentry_context_for_chat(user, session_id, workflow_id, thread_id)

    agent = await create_nexus_chat_agent(
        model=config.default_llm_model,
        checkpointer=checkpointer,
        user_id=user.user_id,
        workflow_id=workflow.workflow_id,  # Public workflow ID for pre-bound tools
        organization_id=user.active_organization_id,
    )
    max_agent_steps, runtime_context = await _get_user_settings_and_context(
        user, thread_id
    )

    return ChatExecutionContext(
        user=user,
        checkpointer=checkpointer,
        thread_id=thread_id,
        agent=agent,
        max_agent_steps=max_agent_steps,
        runtime_context=runtime_context,
    )


async def _finalize_interrupted_chat_execution(
    session: WorkflowChatSession,
    result: Dict[str, Any],
    interrupt_data: Dict[str, Any],
    session_id: int,
    publisher: StreamPublisher,
) -> None:
    """Persist interrupt state and notify the client."""
    logger.info(
        "Chat execution interrupted for user input",
        extra={"session_id": session_id, "interrupt_type": interrupt_data.get("type")},
    )

    session.current_execution_status = ChatExecutionStatus.INTERRUPTED
    session.current_execution_finished_at = datetime.now(timezone.utc)
    session.current_execution_task_id = None
    session.pending_interrupt_type = interrupt_data.get("type")
    session.pending_interrupt_data = interrupt_data
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_task_id",
            "pending_interrupt_type",
            "pending_interrupt_data",
        ]
    )

    response_text = _extract_response_text(result)
    agent_messages = result.get("messages", [])
    thinking_steps = extract_thinking_from_messages(agent_messages)
    await save_chat_message(
        session_id=session_id,
        role="assistant",
        content=response_text,
        thinking="\n".join(thinking_steps) if thinking_steps else None,
    )

    await publisher.publish(StreamEventType.INTERRUPT, interrupt_data)
    await publisher.publish_done()


async def _finalize_completed_chat_execution(
    session: WorkflowChatSession,
    result: Dict[str, Any],
    thread_id: str,
    session_id: int,
    publisher: StreamPublisher,
) -> None:
    """Persist completion state and notify the client."""
    agent_messages = result.get("messages", [])
    response_text = result.get("final_content") or _extract_response_text(result)
    thinking_steps = extract_thinking_from_messages(agent_messages)

    proposal = await WorkflowProposal.get_or_none(
        thread_id=thread_id,
        status=WorkflowProposal.STATUS_PENDING,
    ).prefetch_related("created_by", "workflow", "session")

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
    session.current_execution_task_id = None
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_error",
            "current_execution_task_id",
        ]
    )

    logger.info(
        "Chat execution completed successfully", extra={"session_id": session_id}
    )

    agent_end_data: Dict[str, Any] = {"content": response_text}
    if proposal:
        agent_end_data["proposal_id"] = proposal.id
    await publisher.publish(StreamEventType.AGENT_END, agent_end_data)
    await publisher.publish_done()

    if config.memory_enabled and config.memory_extraction_enabled:
        await extract_session_memories.kiq(session_id)


async def _process_chat_execution_result(
    result: Dict[str, Any],
    context: ChatExecutionContext,
    session_id: int,
    execution_task_id: Optional[str],
    publisher: StreamPublisher,
) -> None:
    """Persist the final execution outcome for a streamed chat run."""
    interrupt_required, interrupt_data = InterruptHandler.extract_interrupt_from_result(
        result
    )
    if not interrupt_required:
        (
            interrupt_required,
            interrupt_data,
        ) = await InterruptHandler.extract_interrupt_from_state(
            context.agent, {"configurable": {"thread_id": context.thread_id}}
        )

    phase = (
        "save_interrupt" if interrupt_required and interrupt_data else "save_completion"
    )
    session = await _get_session_if_current_owner(session_id, execution_task_id, phase)
    if session is None:
        return

    if interrupt_required and interrupt_data:
        await _finalize_interrupted_chat_execution(
            session, result, interrupt_data, session_id, publisher
        )
        return

    await _finalize_completed_chat_execution(
        session, result, context.thread_id, session_id, publisher
    )


async def _handle_chat_execution_cost_cap(
    session_id: int,
    execution_task_id: Optional[str],
    publisher: StreamPublisher,
    error: RunCostCapExceeded,
) -> None:
    """Persist cost-cap failures for chat execution."""
    current_session = await _get_session_if_current_owner(
        session_id, execution_task_id, "cost_cap_exceeded"
    )
    if current_session is None:
        return

    logger.warning(
        "Chat cost cap exceeded",
        extra={
            "session_id": session_id,
            "accumulated_cost": error.accumulated_cost,
            "cost_cap": error.cost_cap,
        },
    )
    current_session.current_execution_status = ChatExecutionStatus.FAILED
    current_session.current_execution_finished_at = datetime.now(timezone.utc)
    current_session.current_execution_error = {
        "type": "cost_cap_exceeded",
        "detail": str(error.to_dict()),
        "status": 402,
    }
    current_session.current_execution_task_id = None
    await current_session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_error",
            "current_execution_task_id",
        ]
    )
    await publisher.publish(
        StreamEventType.ERROR, {"status_code": 402, "message": str(error)}
    )
    await publisher.publish_done()


async def _run_chat_execution_stream(
    session_id: int,
    message: str,
    execution_task_id: Optional[str],
    context: ChatExecutionContext,
) -> None:
    """Run the streaming portion of chat execution."""
    set_chat_runtime_context(context.runtime_context)

    publisher = StreamPublisher(session_id)
    stream_started = asyncio.get_running_loop().time()
    try:
        session = await _get_session_if_current_owner(
            session_id, execution_task_id, "before_stream_start"
        )
        if session is None:
            return

        await publisher.publish(StreamEventType.AGENT_START, {})

        user_msg = HumanMessage(content=message)
        with langfuse_user_context(context.user.user_id):
            result = await _stream_agent_with_orchestrator(
                context.agent,
                context.checkpointer,
                user_msg,
                context.thread_id,
                context.max_agent_steps,
                publisher,
            )

        await _process_chat_execution_result(
            result=result,
            context=context,
            session_id=session_id,
            execution_task_id=execution_task_id,
            publisher=publisher,
        )
        logger.info(
            "Chat execution stream finished",
            extra={
                "session_id": session_id,
                "execution_task_id": execution_task_id,
                "stream_duration_seconds": round(
                    asyncio.get_running_loop().time() - stream_started, 3
                ),
            },
        )
    except RunCostCapExceeded as error:
        await _handle_chat_execution_cost_cap(
            session_id, execution_task_id, publisher, error
        )
    finally:
        clear_chat_runtime_context()
        await publisher.close()


async def _stream_chat_resume_events(
    agent,
    resume_command: Command,
    config_with_langfuse: Dict[str, Any],
    publisher: StreamPublisher,
) -> Dict[str, Any]:
    """Stream resume events and collect the final agent result."""
    final_content = ""
    all_messages: list = []

    async for event in agent.astream_events(
        resume_command, config=config_with_langfuse, version="v2"
    ):
        event_final_content, event_messages = await process_stream_event(
            event, publisher, StreamEventType
        )
        if event_final_content is not None:
            final_content = event_final_content
        if event_messages is not None:
            all_messages = event_messages

    return {"messages": all_messages, "final_content": final_content}


async def _finalize_interrupted_chat_resume(
    session: WorkflowChatSession,
    result: Dict[str, Any],
    interrupt_data: Dict[str, Any],
    session_id: int,
    publisher: StreamPublisher,
) -> None:
    """Persist resumed execution interrupt state and notify the client."""
    logger.info(
        "Chat resume interrupted again for user input",
        extra={"session_id": session_id, "interrupt_type": interrupt_data.get("type")},
    )

    session.current_execution_status = ChatExecutionStatus.INTERRUPTED
    session.current_execution_finished_at = datetime.now(timezone.utc)
    session.current_execution_task_id = None
    session.pending_interrupt_type = interrupt_data.get("type")
    session.pending_interrupt_data = interrupt_data
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_task_id",
            "pending_interrupt_type",
            "pending_interrupt_data",
        ]
    )

    response_text = result.get("final_content") or _extract_response_text(result)
    agent_messages = result.get("messages", [])
    thinking_steps = extract_thinking_from_messages(agent_messages)
    await save_chat_message(
        session_id=session_id,
        role="assistant",
        content=response_text,
        thinking="\n".join(thinking_steps) if thinking_steps else None,
    )

    await publisher.publish(StreamEventType.INTERRUPT, interrupt_data)
    await publisher.publish_done()


async def _finalize_completed_chat_resume(
    session: WorkflowChatSession,
    result: Dict[str, Any],
    session_id: int,
    execution_task_id: Optional[str],
    publisher: StreamPublisher,
) -> None:
    """Persist resumed execution completion state and notify the client."""
    agent_messages = result.get("messages", [])
    response_text = result.get("final_content") or _extract_response_text(result)
    thinking_steps = extract_thinking_from_messages(agent_messages)

    proposal = await WorkflowProposal.get_or_none(
        thread_id=session.thread_id,
        status=WorkflowProposal.STATUS_PENDING,
    ).prefetch_related("created_by", "workflow", "session")

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
    session.current_execution_task_id = None
    await session.save(
        update_fields=[
            "current_execution_status",
            "current_execution_finished_at",
            "current_execution_error",
            "current_execution_task_id",
        ]
    )

    logger.info(
        "Chat resume completed successfully",
        extra={
            "session_id": session_id,
            "execution_task_id": execution_task_id,
            "thread_id": session.thread_id,
        },
    )

    agent_end_data: Dict[str, Any] = {"content": response_text}
    if proposal:
        agent_end_data["proposal_id"] = proposal.id
    await publisher.publish(StreamEventType.AGENT_END, agent_end_data)
    await publisher.publish_done()


async def _process_chat_resume_result(
    result: Dict[str, Any],
    context: ChatExecutionContext,
    session_id: int,
    execution_task_id: Optional[str],
    publisher: StreamPublisher,
) -> None:
    """Persist the final outcome for a resumed chat execution."""
    interrupt_required, interrupt_data = InterruptHandler.extract_interrupt_from_result(
        result
    )
    if not interrupt_required:
        (
            interrupt_required,
            interrupt_data,
        ) = await InterruptHandler.extract_interrupt_from_state(
            context.agent,
            {"configurable": {"thread_id": context.thread_id}},
        )

    phase = (
        "save_resume_interrupt"
        if interrupt_required and interrupt_data
        else "save_resume_completion"
    )
    session = await _get_session_if_current_owner(session_id, execution_task_id, phase)
    if session is None:
        return

    if interrupt_required and interrupt_data:
        await _finalize_interrupted_chat_resume(
            session, result, interrupt_data, session_id, publisher
        )
        return

    await _finalize_completed_chat_resume(
        session=session,
        result=result,
        session_id=session_id,
        execution_task_id=execution_task_id,
        publisher=publisher,
    )


async def _run_chat_resume_stream(
    session_id: int,
    resume_command_data: Dict[str, Any],
    execution_task_id: Optional[str],
    context: ChatExecutionContext,
) -> None:
    """Run the streaming portion of a resumed chat execution."""
    set_chat_runtime_context(context.runtime_context)

    publisher = StreamPublisher(session_id)
    token = _current_thread_id.set(context.thread_id)
    attempt_token = _submission_attempt_count.set(0)
    stream_started = asyncio.get_running_loop().time()
    try:
        session = await _get_session_if_current_owner(
            session_id, execution_task_id, "before_resume_stream_start"
        )
        if session is None:
            return

        await publisher.publish(StreamEventType.AGENT_START, {})

        cost_callback = CostCapCallbackHandler()
        config_with_langfuse = merge_nexus_langfuse_callbacks(
            {
                "configurable": {"thread_id": context.thread_id},
                "recursion_limit": context.max_agent_steps,
                "callbacks": [cost_callback],
            }
        )
        resume_command = Command(resume=resume_command_data)

        with langfuse_user_context(context.user.user_id):
            try:
                result = await asyncio.wait_for(
                    _stream_chat_resume_events(
                        context.agent, resume_command, config_with_langfuse, publisher
                    ),
                    timeout=float(config.nexus_chat_timeout_seconds),
                )
            except TimeoutError:
                logger.error(
                    "Chat resume stream timed out",
                    extra={
                        "session_id": session_id,
                        "execution_task_id": execution_task_id,
                        "thread_id": context.thread_id,
                        "stream_duration_seconds": round(
                            asyncio.get_running_loop().time() - stream_started, 3
                        ),
                        "timeout_seconds": float(config.nexus_chat_timeout_seconds),
                    },
                )
                raise

        logger.info(
            "Chat resume stream finished",
            extra={
                "session_id": session_id,
                "execution_task_id": execution_task_id,
                "thread_id": context.thread_id,
                "stream_duration_seconds": round(
                    asyncio.get_running_loop().time() - stream_started, 3
                ),
                "timeout_seconds": float(config.nexus_chat_timeout_seconds),
            },
        )

        await _process_chat_resume_result(
            result=result,
            context=context,
            session_id=session_id,
            execution_task_id=execution_task_id,
            publisher=publisher,
        )
    finally:
        _current_thread_id.reset(token)
        _submission_attempt_count.reset(attempt_token)
        clear_chat_runtime_context()
        await publisher.close()


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


async def _stream_agent_with_orchestrator(  # pylint: disable=too-many-positional-arguments # Reason: All params required for streaming orchestration
    agent,
    checkpointer,
    user_msg: HumanMessage,
    thread_id: str,
    max_agent_steps: int,
    publisher: StreamPublisher,
) -> Dict[str, Any]:
    """Stream agent events using orchestrator with health checks."""
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

    return await orchestrator.stream_with_health_checks(
        user_msg, config_dict, publisher
    )


@broker.task
async def chat_execution_task(
    # pylint: disable=too-many-positional-arguments
    # Reason: Background task requires all parameters from the broker payload
    session_id: int,
    user_id: int,
    message: str,
    workflow_id: int,
    model: Optional[str] = None,
    execution_task_id: Optional[str] = None,
    user_timezone: Optional[str] = None,
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
            "execution_task_id": execution_task_id,
        },
    )

    request = ChatExecutionRequest(
        session_id=session_id,
        user_id=user_id,
        message=message,
        workflow_id=workflow_id,
        execution_task_id=execution_task_id,
        model=model,
        timezone=user_timezone,
    )

    lock: Optional[
        _ChatExecutionSingleFlightLock | _NoOpChatExecutionSingleFlightLock
    ] = None

    try:
        session = await _get_session_if_current_owner(
            session_id, execution_task_id, "start"
        )
        if session is None or _should_abort_chat_execution(
            session, session_id, execution_task_id
        ):
            return

        lock = await _acquire_execution_single_flight_lock(
            session_id, execution_task_id, "chat_execution_task"
        )
        if lock is None:
            return

        session = await _get_session_if_current_owner(
            session_id, execution_task_id, "start_locked"
        )
        if session is None or _should_abort_chat_execution(
            session, session_id, execution_task_id
        ):
            return

        await _mark_chat_execution_running(session)

        context = await _prepare_chat_execution(request)
        if context is None:
            return

        await _run_chat_execution_stream(
            session_id=session_id,
            message=message,
            execution_task_id=execution_task_id,
            context=context,
        )

    except asyncio.CancelledError as e:
        error_payload = _build_cancelled_execution_error_payload(e)
        logger.warning(
            "Chat execution task cancelled",
            exc_info=True,
            extra={
                "session_id": session_id,
                "execution_task_id": execution_task_id,
                "error": error_payload["detail"],
                "error_type": error_payload["type"],
                "exception_type": error_payload["exception_type"],
            },
        )
        await _persist_failure_if_current_owner(
            session_id, execution_task_id, error_payload
        )
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Background task must catch all exceptions to avoid worker crash
        error_payload = _build_execution_error_payload(e)
        logger.error(
            "Chat execution task failed",
            exc_info=True,
            extra={
                "session_id": session_id,
                "execution_task_id": execution_task_id,
                "error": error_payload["detail"],
                "error_type": error_payload["type"],
                "exception_type": error_payload["exception_type"],
            },
        )
        await _persist_failure_if_current_owner(
            session_id, execution_task_id, error_payload
        )
    finally:
        if lock is not None:
            await lock.release()


@broker.task
async def chat_resume_task(
    # pylint: disable=too-many-positional-arguments,too-many-locals,too-many-statements,too-complex
    # Reason: Background task requires all parameters; resume orchestration has inherent branching
    session_id: int,
    user_id: int,
    thread_id: str,
    resume_command_data: Dict[str, Any],
    workflow_id: int,
    execution_task_id: Optional[str] = None,
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
            "execution_task_id": execution_task_id,
        },
    )

    lock: Optional[
        _ChatExecutionSingleFlightLock | _NoOpChatExecutionSingleFlightLock
    ] = None

    try:
        session = await _get_session_if_current_owner(
            session_id, execution_task_id, "resume_start"
        )
        if session is None:
            return

        lock = await _acquire_execution_single_flight_lock(
            session_id, execution_task_id, "chat_resume_task"
        )
        if lock is None:
            return

        session = await _get_session_if_current_owner(
            session_id, execution_task_id, "resume_start_locked"
        )
        if session is None:
            return

        await _mark_chat_resume_running(session)

        context = await _prepare_chat_resume_execution(
            session_id=session_id,
            user_id=user_id,
            thread_id=thread_id,
            workflow_id=workflow_id,
            execution_task_id=execution_task_id,
        )
        if context is None:
            return

        await _run_chat_resume_stream(
            session_id=session_id,
            resume_command_data=resume_command_data,
            execution_task_id=execution_task_id,
            context=context,
        )

    except asyncio.CancelledError as e:
        error_payload = _build_cancelled_execution_error_payload(e)
        logger.warning(
            "Chat resume task cancelled",
            exc_info=True,
            extra={
                "session_id": session_id,
                "execution_task_id": execution_task_id,
                "error": error_payload["detail"],
                "error_type": error_payload["type"],
                "exception_type": error_payload["exception_type"],
            },
        )
        await _persist_failure_if_current_owner(
            session_id, execution_task_id, error_payload
        )
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Background task must catch all exceptions to avoid worker crash
        error_payload = _build_execution_error_payload(e)
        logger.error(
            "Chat resume task failed",
            exc_info=True,
            extra={
                "session_id": session_id,
                "execution_task_id": execution_task_id,
                "error": error_payload["detail"],
                "error_type": error_payload["type"],
                "exception_type": error_payload["exception_type"],
            },
        )
        await _persist_failure_if_current_owner(
            session_id, execution_task_id, error_payload
        )
    finally:
        if lock is not None:
            await lock.release()


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
            ChatExecutionStatus.RUNNING,
        ],
        current_execution_started_at__lt=cutoff_time,
    ).all()

    for session in stale_sessions:
        session.current_execution_status = ChatExecutionStatus.FAILED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_task_id = None
        session.current_execution_error = {
            "type": "timeout",
            "detail": "Execution timeout - session was stale",
            "reason": "cleanup_task",
            "status": 500,
        }
        await session.save(
            update_fields=[
                "current_execution_status",
                "current_execution_finished_at",
                "current_execution_error",
                "current_execution_task_id",
            ]
        )

    logger.info("Cleaned up %d stale chat executions", len(stale_sessions))


__all__ = [
    "chat_execution_task",
    "chat_resume_task",
    "cleanup_stale_chat_executions",
    "_handle_agent_result",
]
