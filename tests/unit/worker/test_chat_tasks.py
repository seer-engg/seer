"""
Unit tests for worker.tasks.chat module.

Tests pure logic functions that don't require database or external services:
- Error payload builders
- Upstream stream error detection
- Exception chain iteration
- Session ownership checks
- Lock configuration computation
- Chat execution abort logic
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from seer.worker.tasks.chat import (
    ChatExecutionRequest,
    _build_cancelled_execution_error_payload,
    _build_execution_error_payload,
    _ChatExecutionLockConfig,
    _ChatExecutionSingleFlightLock,
    _extract_response_text,
    _is_upstream_stream_error,
    _iter_exception_chain,
    _NoOpChatExecutionSingleFlightLock,
    _should_abort_chat_execution,
    _TERMINAL_CHAT_STATUSES,
)
from seer.database.workflow_models import ChatExecutionStatus


# =============================================================================
# Error Payload Builders
# =============================================================================


@pytest.mark.unit
class TestBuildExecutionErrorPayload:
    """Tests for _build_execution_error_payload."""

    def test_generic_exception(self):
        error = ValueError("Something went wrong")
        payload = _build_execution_error_payload(error)

        assert payload["type"] == "execution_error"
        assert payload["detail"] == "Something went wrong"
        assert payload["status"] == 500
        assert "ValueError" in payload["exception_type"]

    def test_empty_message_uses_class_name(self):
        error = RuntimeError("")
        payload = _build_execution_error_payload(error)

        assert payload["detail"] == "RuntimeError"

    def test_upstream_stream_error_returns_502(self):
        error = httpx.ReadTimeout("Connection timed out")
        payload = _build_execution_error_payload(error)

        assert payload["type"] == "upstream_stream_error"
        assert payload["status"] == 502
        assert "exception_detail" in payload

    def test_upstream_error_with_cause_chain(self):
        """Upstream error nested in a cause chain should still be detected."""
        inner = httpx.ConnectError("Connection refused")
        outer = RuntimeError("Agent execution failed")
        outer.__cause__ = inner

        payload = _build_execution_error_payload(outer)
        assert payload["type"] == "upstream_stream_error"
        assert payload["status"] == 502

    def test_non_upstream_error_returns_500(self):
        error = KeyError("missing_key")
        payload = _build_execution_error_payload(error)

        assert payload["type"] == "execution_error"
        assert payload["status"] == 500


@pytest.mark.unit
class TestBuildCancelledExecutionErrorPayload:
    """Tests for _build_cancelled_execution_error_payload."""

    def test_cancelled_error_payload(self):
        import asyncio
        error = asyncio.CancelledError()
        payload = _build_cancelled_execution_error_payload(error)

        assert payload["type"] == "execution_cancelled"
        assert payload["status"] == 503
        assert "cancelled" in payload["detail"].lower()
        assert "CancelledError" in payload["exception_type"]


# =============================================================================
# Upstream Stream Error Detection
# =============================================================================


@pytest.mark.unit
class TestIsUpstreamStreamError:
    """Tests for _is_upstream_stream_error."""

    def test_httpx_error(self):
        assert _is_upstream_stream_error(httpx.ReadTimeout("timeout")) is True

    def test_httpx_connect_error(self):
        assert _is_upstream_stream_error(httpx.ConnectError("refused")) is True

    def test_generic_exception_is_not_upstream(self):
        assert _is_upstream_stream_error(ValueError("nope")) is False

    def test_runtime_error_is_not_upstream(self):
        assert _is_upstream_stream_error(RuntimeError("something")) is False

    def test_httpx_error_in_cause_chain(self):
        inner = httpx.ReadTimeout("timed out")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert _is_upstream_stream_error(outer) is True

    def test_openai_connection_error(self):
        """OpenAI APIConnectionError should be detected as upstream."""
        # Create a custom exception class that mimics openai.APIConnectionError
        OpenAIConnectionError = type(
            "APIConnectionError",
            (Exception,),
            {"__module__": "openai"},
        )
        error = OpenAIConnectionError("connection failed")
        assert _is_upstream_stream_error(error) is True

    def test_deep_cause_chain(self):
        """Error nested 3 levels deep should still be detected."""
        inner = httpx.ReadTimeout("timeout")
        middle = RuntimeError("middle")
        middle.__cause__ = inner
        outer = ValueError("outer")
        outer.__cause__ = middle
        assert _is_upstream_stream_error(outer) is True


# =============================================================================
# Exception Chain Iteration
# =============================================================================


@pytest.mark.unit
class TestIterExceptionChain:
    """Tests for _iter_exception_chain."""

    def test_single_exception(self):
        e = ValueError("test")
        chain = _iter_exception_chain(e)
        assert len(chain) == 1
        assert chain[0] is e

    def test_cause_chain(self):
        inner = TypeError("inner")
        outer = ValueError("outer")
        outer.__cause__ = inner
        chain = _iter_exception_chain(outer)
        assert len(chain) == 2
        assert chain[0] is outer
        assert chain[1] is inner

    def test_context_chain(self):
        inner = TypeError("inner")
        outer = ValueError("outer")
        outer.__context__ = inner
        chain = _iter_exception_chain(outer)
        assert len(chain) == 2

    def test_circular_reference_does_not_infinite_loop(self):
        """Circular cause chains should be detected via seen set."""
        a = ValueError("a")
        b = TypeError("b")
        a.__cause__ = b
        b.__cause__ = a
        chain = _iter_exception_chain(a)
        assert len(chain) == 2


# =============================================================================
# Response Text Extraction
# =============================================================================


@pytest.mark.unit
class TestExtractResponseText:
    """Tests for _extract_response_text."""

    def test_with_messages(self):
        msg = MagicMock()
        msg.content = "Hello, how can I help?"
        result = {"messages": [msg]}
        assert _extract_response_text(result) == "Hello, how can I help?"

    def test_with_multiple_messages_takes_last(self):
        msg1 = MagicMock()
        msg1.content = "First"
        msg2 = MagicMock()
        msg2.content = "Last response"
        result = {"messages": [msg1, msg2]}
        assert _extract_response_text(result) == "Last response"

    def test_empty_messages(self):
        result = {"messages": []}
        text = _extract_response_text(result)
        assert "help" in text.lower()

    def test_none_result(self):
        text = _extract_response_text(None)
        assert "help" in text.lower()

    def test_message_without_content_attr(self):
        result = {"messages": ["raw string"]}
        text = _extract_response_text(result)
        assert text == "raw string"


# =============================================================================
# Session Ownership & Abort Logic
# =============================================================================


@pytest.mark.unit
class TestShouldAbortChatExecution:
    """Tests for _should_abort_chat_execution."""

    def test_interrupted_session_aborts(self):
        session = MagicMock()
        session.current_execution_status = ChatExecutionStatus.INTERRUPTED
        session.pending_interrupt_data = None
        assert _should_abort_chat_execution(session, 1, "task-1") is True

    def test_session_with_pending_interrupt_aborts(self):
        session = MagicMock()
        session.current_execution_status = ChatExecutionStatus.RUNNING
        session.pending_interrupt_data = {"type": "human_input"}
        assert _should_abort_chat_execution(session, 1, "task-1") is True

    def test_normal_session_does_not_abort(self):
        session = MagicMock()
        session.current_execution_status = ChatExecutionStatus.QUEUED
        session.pending_interrupt_data = None
        assert _should_abort_chat_execution(session, 1, "task-1") is False


# =============================================================================
# Terminal Status Constants
# =============================================================================


@pytest.mark.unit
class TestTerminalStatuses:
    """Tests for _TERMINAL_CHAT_STATUSES."""

    def test_completed_is_terminal(self):
        assert ChatExecutionStatus.COMPLETED in _TERMINAL_CHAT_STATUSES

    def test_failed_is_terminal(self):
        assert ChatExecutionStatus.FAILED in _TERMINAL_CHAT_STATUSES

    def test_interrupted_is_terminal(self):
        assert ChatExecutionStatus.INTERRUPTED in _TERMINAL_CHAT_STATUSES

    def test_running_is_not_terminal(self):
        assert ChatExecutionStatus.RUNNING not in _TERMINAL_CHAT_STATUSES

    def test_queued_is_not_terminal(self):
        assert ChatExecutionStatus.QUEUED not in _TERMINAL_CHAT_STATUSES


# =============================================================================
# Lock Configuration
# =============================================================================


@pytest.mark.unit
class TestChatExecutionLockConfig:
    """Tests for _ChatExecutionSingleFlightLock configuration."""

    def test_lock_key_format(self):
        with patch("seer.worker.tasks.chat.config") as mock_config:
            mock_config.nexus_chat_timeout_seconds = 900
            lock = _ChatExecutionSingleFlightLock(
                session_id=42,
                execution_task_id="task-abc",
                task_name="chat_execution_task",
            )
        assert "42" in lock.lock_key
        assert "task-abc" in lock.lock_key
        assert lock.lock_key.startswith("nexus:chat-execution:")

    def test_no_execution_task_id_uses_no_owner(self):
        with patch("seer.worker.tasks.chat.config") as mock_config:
            mock_config.nexus_chat_timeout_seconds = 900
            lock = _ChatExecutionSingleFlightLock(
                session_id=1,
                execution_task_id=None,
                task_name="test",
            )
        assert "no-owner" in lock.lock_key

    def test_ttl_respects_minimum(self):
        """TTL should never go below _CHAT_LOCK_MIN_TTL_MS."""
        with patch("seer.worker.tasks.chat.config") as mock_config:
            mock_config.nexus_chat_timeout_seconds = 1  # Very short timeout
            lock = _ChatExecutionSingleFlightLock(
                session_id=1,
                execution_task_id="t",
                task_name="test",
            )
        assert lock._config.ttl_ms >= 30000  # _CHAT_LOCK_MIN_TTL_MS


@pytest.mark.unit
class TestNoOpLock:
    """Tests for _NoOpChatExecutionSingleFlightLock."""

    @pytest.mark.asyncio
    async def test_release_is_noop(self):
        lock = _NoOpChatExecutionSingleFlightLock(session_id=1, task_name="test")
        await lock.release()  # Should not raise

    def test_lock_key_is_empty(self):
        lock = _NoOpChatExecutionSingleFlightLock(session_id=1, task_name="test")
        assert lock.lock_key == ""


# =============================================================================
# ChatExecutionRequest Dataclass
# =============================================================================


@pytest.mark.unit
class TestChatExecutionRequest:
    """Tests for ChatExecutionRequest dataclass."""

    def test_required_fields(self):
        req = ChatExecutionRequest(
            session_id=1,
            user_id=2,
            message="Hello",
            workflow_id=3,
            execution_task_id=None,
            model=None,
        )
        assert req.session_id == 1
        assert req.user_id == 2
        assert req.message == "Hello"
        assert req.workflow_id == 3

    def test_optional_fields(self):
        req = ChatExecutionRequest(
            session_id=1,
            user_id=2,
            message="Hello",
            workflow_id=3,
            execution_task_id="task-123",
            model="claude-3-haiku",
        )
        assert req.execution_task_id == "task-123"
        assert req.model == "claude-3-haiku"


# =============================================================================
# Session Ownership Checks
# =============================================================================


@pytest.mark.unit
class TestGetSessionIfCurrentOwner:
    """Tests for _get_session_if_current_owner."""

    @pytest.mark.asyncio
    async def test_returns_session_when_owner_matches(self):
        from seer.worker.tasks.chat import _get_session_if_current_owner

        session = MagicMock()
        session.current_execution_task_id = "task-1"

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=session)
            result = await _get_session_if_current_owner(1, "task-1", "test")

        assert result is session

    @pytest.mark.asyncio
    async def test_returns_none_when_session_not_found(self):
        from seer.worker.tasks.chat import _get_session_if_current_owner

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=None)
            result = await _get_session_if_current_owner(1, "task-1", "test")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_owner_mismatch(self):
        from seer.worker.tasks.chat import _get_session_if_current_owner

        session = MagicMock()
        session.current_execution_task_id = "task-2"  # Different owner

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=session)
            result = await _get_session_if_current_owner(1, "task-1", "test")

        assert result is None

    @pytest.mark.asyncio
    async def test_no_execution_task_id_skips_ownership_check(self):
        from seer.worker.tasks.chat import _get_session_if_current_owner

        session = MagicMock()
        session.current_execution_task_id = "anything"

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=session)
            result = await _get_session_if_current_owner(1, None, "test")

        assert result is session


# =============================================================================
# Persist Failure If Current Owner
# =============================================================================


@pytest.mark.unit
class TestPersistFailureIfCurrentOwner:
    """Tests for _persist_failure_if_current_owner."""

    @pytest.mark.asyncio
    async def test_persists_failure_when_owner_matches(self):
        from seer.worker.tasks.chat import _persist_failure_if_current_owner

        session = MagicMock()
        session.current_execution_task_id = "task-1"
        session.current_execution_status = ChatExecutionStatus.RUNNING
        session.save = AsyncMock()

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=session)
            await _persist_failure_if_current_owner(1, "task-1", {"type": "error"})

        assert session.current_execution_status == ChatExecutionStatus.FAILED
        session.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_if_already_terminal(self):
        from seer.worker.tasks.chat import _persist_failure_if_current_owner

        session = MagicMock()
        session.current_execution_task_id = "task-1"
        session.current_execution_status = ChatExecutionStatus.COMPLETED
        session.save = AsyncMock()

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=session)
            await _persist_failure_if_current_owner(1, "task-1", {"type": "error"})

        session.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_if_session_not_found(self):
        from seer.worker.tasks.chat import _persist_failure_if_current_owner

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_model.get_or_none = AsyncMock(return_value=None)
            # Should not raise
            await _persist_failure_if_current_owner(1, "task-1", {"type": "error"})


# =============================================================================
# Cleanup Stale Chat Executions
# =============================================================================


@pytest.mark.unit
class TestCleanupStaleChatExecutions:
    """Tests for cleanup_stale_chat_executions task."""

    @pytest.mark.asyncio
    async def test_resets_stale_sessions_to_failed(self):
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        stale_session = MagicMock()
        stale_session.save = AsyncMock()

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_qs = MagicMock()
            mock_qs.all = AsyncMock(return_value=[stale_session])
            mock_model.filter.return_value = mock_qs

            await cleanup_stale_chat_executions()

        assert stale_session.current_execution_status == ChatExecutionStatus.FAILED
        assert stale_session.current_execution_error["type"] == "timeout"
        stale_session.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_stale_sessions(self):
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as mock_model:
            mock_qs = MagicMock()
            mock_qs.all = AsyncMock(return_value=[])
            mock_model.filter.return_value = mock_qs

            # Should not raise
            await cleanup_stale_chat_executions()
