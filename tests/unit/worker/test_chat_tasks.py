"""
Unit tests for worker.tasks.chat module.

Tests background chat execution tasks including status transitions,
error handling, and interrupt detection.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.database.workflow_models import ChatExecutionStatus


async def _empty_astream_events(*args, **kwargs):
    """Async generator stub that yields nothing — prevents asyncio.wait_for(timeout=900) hang."""
    return
    yield  # noqa: unreachable — makes this an async generator


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
class TestSetSentryContextForChat:
    """Tests for _set_sentry_context_for_chat function."""

    def test_sets_basic_tags(self, mock_user):
        """Test that basic Sentry tags are set."""
        from seer.worker.tasks.chat import _set_sentry_context_for_chat

        with patch("seer.worker.tasks.chat.set_tag") as mock_set_tag, \
             patch("seer.worker.tasks.chat.set_user_context"), \
             patch("seer.worker.tasks.chat.set_context"):

            _set_sentry_context_for_chat(mock_user, session_id=1, workflow_id=2, thread_id="t123")

        # Verify tags were set
        calls = {call.args[0]: call.args[1] for call in mock_set_tag.call_args_list}
        assert calls["task_type"] == "chat_execution"
        assert calls["session_id"] == "1"
        assert calls["workflow_id"] == "2"

    def test_sets_user_context(self, mock_user):
        """Test that user context is set with correct attributes."""
        from seer.worker.tasks.chat import _set_sentry_context_for_chat

        mock_user.first_name = "John"
        mock_user.last_name = "Doe"

        with patch("seer.worker.tasks.chat.set_tag"), \
             patch("seer.worker.tasks.chat.set_user_context") as mock_set_user, \
             patch("seer.worker.tasks.chat.set_context"):

            _set_sentry_context_for_chat(mock_user, session_id=1, workflow_id=2)

        mock_set_user.assert_called_once()
        call_kwargs = mock_set_user.call_args.kwargs
        assert call_kwargs["user_id"] == mock_user.user_id
        assert call_kwargs["email"] == mock_user.email

    def test_sets_chat_session_context(self, mock_user):
        """Test that chat session context is set."""
        from seer.worker.tasks.chat import _set_sentry_context_for_chat

        with patch("seer.worker.tasks.chat.set_tag"), \
             patch("seer.worker.tasks.chat.set_user_context"), \
             patch("seer.worker.tasks.chat.set_context") as mock_set_context:

            _set_sentry_context_for_chat(mock_user, session_id=5, workflow_id=10, thread_id="thread-abc")

        mock_set_context.assert_called_once_with("chat_session", {
            "session_id": 5,
            "workflow_id": 10,
            "thread_id": "thread-abc",
        })

    def test_silently_handles_exception(self, mock_user):
        """Test that exceptions in Sentry setup don't propagate."""
        from seer.worker.tasks.chat import _set_sentry_context_for_chat

        with patch("seer.worker.tasks.chat.set_tag"), \
             patch("seer.worker.tasks.chat.set_user_context", side_effect=RuntimeError("Sentry error")), \
             patch("seer.worker.tasks.chat.set_context"), \
             patch("seer.worker.tasks.chat.logger") as mock_logger:

            # Should not raise
            _set_sentry_context_for_chat(mock_user, session_id=1, workflow_id=2)

        mock_logger.debug.assert_called()

    def test_handles_missing_user_attributes(self):
        """Test graceful handling when user lacks optional attributes."""
        from seer.worker.tasks.chat import _set_sentry_context_for_chat

        # User without email or name attributes
        user = MagicMock()
        user.user_id = "user_456"
        del user.email
        del user.first_name
        del user.last_name

        with patch("seer.worker.tasks.chat.set_tag"), \
             patch("seer.worker.tasks.chat.set_user_context") as mock_set_user, \
             patch("seer.worker.tasks.chat.set_context"):

            # Should not raise
            _set_sentry_context_for_chat(user, session_id=1, workflow_id=2)

        # Should still be called with None for missing attributes
        mock_set_user.assert_called_once()


@pytest.mark.unit
class TestExtractResponseText:
    """Tests for _extract_response_text function."""

    def test_extracts_from_message_content(self):
        """Test extracting content from last message."""
        from seer.worker.tasks.chat import _extract_response_text

        last_msg = MagicMock()
        last_msg.content = "Hello, how can I help?"

        result = _extract_response_text({"messages": [last_msg]})

        assert result == "Hello, how can I help?"

    def test_returns_default_for_empty_messages(self):
        """Test default response when messages list is empty."""
        from seer.worker.tasks.chat import _extract_response_text

        result = _extract_response_text({"messages": []})

        assert result == "I'm here to help with your workflow!"

    def test_returns_default_for_no_messages_key(self):
        """Test default response when messages key is missing."""
        from seer.worker.tasks.chat import _extract_response_text

        result = _extract_response_text({"other": "data"})

        assert result == "I'm here to help with your workflow!"

    def test_returns_default_for_non_dict(self):
        """Test default response for non-dict input."""
        from seer.worker.tasks.chat import _extract_response_text

        result = _extract_response_text("not a dict")

        assert result == "I'm here to help with your workflow!"

    def test_handles_message_without_content_attr(self):
        """Test string conversion fallback for messages without content."""
        from seer.worker.tasks.chat import _extract_response_text

        result = _extract_response_text({"messages": ["string_message"]})

        assert result == "string_message"


@pytest.mark.unit
class TestGetUserSettingsAndContext:
    """Tests for _get_user_settings_and_context function."""

    @pytest.mark.asyncio
    async def test_returns_user_settings_values(self, mock_user, mock_user_settings):
        """Test returning values from UserSettings."""
        from seer.worker.tasks.chat import _get_user_settings_and_context

        with patch("seer.worker.tasks.chat.UserSettings") as MockSettings:
            MockSettings.get = AsyncMock(return_value=mock_user_settings)

            max_steps, context = await _get_user_settings_and_context(mock_user, "thread-123")

        assert max_steps == 50
        assert context.per_run_cost_cap_usd == 5.0
        assert context.thread_id == "thread-123"
        assert context.user == mock_user

    @pytest.mark.asyncio
    async def test_returns_defaults_when_no_settings(self, mock_user):
        """Test falling back to config defaults when settings not found."""
        from tortoise.exceptions import DoesNotExist

        from seer.database.models import UserSettings
        from seer.worker.tasks.chat import _get_user_settings_and_context

        with patch("seer.worker.tasks.chat.UserSettings") as MockSettings, \
             patch("seer.worker.tasks.chat.config") as mock_config:
            MockSettings.get = AsyncMock(side_effect=DoesNotExist(UserSettings))
            mock_config.nexus_max_agent_steps = 100

            max_steps, context = await _get_user_settings_and_context(mock_user, "thread-456")

        assert max_steps == 100
        assert context.per_run_cost_cap_usd == 5.0

    @pytest.mark.asyncio
    async def test_creates_runtime_context(self, mock_user, mock_user_settings):
        """Test that WorkflowRuntimeContext is created correctly."""
        from seer.worker.tasks.chat import _get_user_settings_and_context

        with patch("seer.worker.tasks.chat.UserSettings") as MockSettings:
            MockSettings.get = AsyncMock(return_value=mock_user_settings)

            _, context = await _get_user_settings_and_context(mock_user, "thread-789")

        assert context.workflow_run_id is None
        assert context.accumulated_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_handles_missing_preferences_key(self, mock_user):
        """Test handling when preferences dict is missing cost cap key."""
        from seer.worker.tasks.chat import _get_user_settings_and_context

        settings = MagicMock()
        settings.max_agent_steps = 25
        settings.preferences = {}  # Missing per_run_cost_cap_usd

        with patch("seer.worker.tasks.chat.UserSettings") as MockSettings:
            MockSettings.get = AsyncMock(return_value=settings)

            _, context = await _get_user_settings_and_context(mock_user, "thread-000")

        assert context.per_run_cost_cap_usd == 5.0  # Default


@pytest.mark.unit
class TestInvokeAgentWithOrchestrator:
    """Tests for _invoke_agent_with_orchestrator function."""

    @pytest.mark.asyncio
    async def test_creates_orchestrator_with_services(self):
        """Test that ChatOrchestrator is created with all services."""
        from seer.worker.tasks.chat import _invoke_agent_with_orchestrator

        mock_agent = MagicMock()
        mock_checkpointer = MagicMock()
        user_msg = MagicMock()

        with patch("seer.worker.tasks.chat.CostCapCallbackHandler"), \
             patch("seer.worker.tasks.chat.ChatOrchestrator") as MockOrchestrator, \
             patch("seer.worker.tasks.chat.CheckpointerHealthService"), \
             patch("seer.worker.tasks.chat.IncompleteToolCallDetector"), \
             patch("seer.worker.tasks.chat.IncompleteToolCallRecoveryService"), \
             patch("seer.worker.tasks.chat._recreate_checkpointer"):

            mock_instance = MagicMock()
            mock_instance.invoke_with_health_checks = AsyncMock(return_value={"messages": []})
            MockOrchestrator.return_value = mock_instance

            await _invoke_agent_with_orchestrator(
                mock_agent, mock_checkpointer, user_msg, "thread-123", 50
            )

        MockOrchestrator.assert_called_once()
        call_kwargs = MockOrchestrator.call_args.kwargs
        assert call_kwargs["agent"] == mock_agent
        assert call_kwargs["checkpointer"] == mock_checkpointer

    @pytest.mark.asyncio
    async def test_invokes_orchestrator_with_message(self):
        """Test that orchestrator.invoke_with_health_checks is called."""
        from seer.worker.tasks.chat import _invoke_agent_with_orchestrator

        mock_agent = MagicMock()
        mock_checkpointer = MagicMock()
        user_msg = MagicMock()

        with patch("seer.worker.tasks.chat.CostCapCallbackHandler"), \
             patch("seer.worker.tasks.chat.ChatOrchestrator") as MockOrchestrator, \
             patch("seer.worker.tasks.chat.CheckpointerHealthService"), \
             patch("seer.worker.tasks.chat.IncompleteToolCallDetector"), \
             patch("seer.worker.tasks.chat.IncompleteToolCallRecoveryService"), \
             patch("seer.worker.tasks.chat._recreate_checkpointer"):

            mock_instance = MagicMock()
            mock_instance.invoke_with_health_checks = AsyncMock(return_value={"result": "ok"})
            MockOrchestrator.return_value = mock_instance

            result = await _invoke_agent_with_orchestrator(
                mock_agent, mock_checkpointer, user_msg, "thread-123", 50
            )

        mock_instance.invoke_with_health_checks.assert_called_once()
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_passes_recursion_limit(self):
        """Test that max_agent_steps is passed as recursion_limit."""
        from seer.worker.tasks.chat import _invoke_agent_with_orchestrator

        mock_agent = MagicMock()
        mock_checkpointer = MagicMock()
        user_msg = MagicMock()

        with patch("seer.worker.tasks.chat.CostCapCallbackHandler"), \
             patch("seer.worker.tasks.chat.ChatOrchestrator") as MockOrchestrator, \
             patch("seer.worker.tasks.chat.CheckpointerHealthService"), \
             patch("seer.worker.tasks.chat.IncompleteToolCallDetector"), \
             patch("seer.worker.tasks.chat.IncompleteToolCallRecoveryService"), \
             patch("seer.worker.tasks.chat._recreate_checkpointer"):

            mock_instance = MagicMock()
            mock_instance.invoke_with_health_checks = AsyncMock(return_value={})
            MockOrchestrator.return_value = mock_instance

            await _invoke_agent_with_orchestrator(
                mock_agent, mock_checkpointer, user_msg, "thread-123", 75
            )

        # Check config_dict passed to invoke_with_health_checks
        call_args = mock_instance.invoke_with_health_checks.call_args
        config_dict = call_args.args[1]
        assert config_dict["recursion_limit"] == 75

    @pytest.mark.asyncio
    async def test_creates_cost_callback(self):
        """Test that CostCapCallbackHandler is created and added to config."""
        from seer.worker.tasks.chat import _invoke_agent_with_orchestrator

        mock_agent = MagicMock()
        mock_checkpointer = MagicMock()
        user_msg = MagicMock()

        with patch("seer.worker.tasks.chat.CostCapCallbackHandler") as MockCallback, \
             patch("seer.worker.tasks.chat.ChatOrchestrator") as MockOrchestrator, \
             patch("seer.worker.tasks.chat.CheckpointerHealthService"), \
             patch("seer.worker.tasks.chat.IncompleteToolCallDetector"), \
             patch("seer.worker.tasks.chat.IncompleteToolCallRecoveryService"), \
             patch("seer.worker.tasks.chat._recreate_checkpointer"):

            mock_callback_instance = MagicMock()
            MockCallback.return_value = mock_callback_instance

            mock_instance = MagicMock()
            mock_instance.invoke_with_health_checks = AsyncMock(return_value={})
            MockOrchestrator.return_value = mock_instance

            await _invoke_agent_with_orchestrator(
                mock_agent, mock_checkpointer, user_msg, "thread-123", 50
            )

        MockCallback.assert_called_once()


# =============================================================================
# Chat Execution Task Tests
# =============================================================================


@contextmanager
def mock_langfuse_context(user_id=None):
    """Context manager mock for langfuse_user_context."""
    yield


@pytest.mark.unit
class TestChatExecutionTask:
    """Tests for chat_execution_task function."""

    @pytest.mark.asyncio
    async def test_sets_status_to_running_on_start(self, mock_user, mock_workflow, mock_chat_session):
        """Test that session status is set to RUNNING when task starts."""
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.extract_session_memories") as mock_extract, \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_stream.return_value = {"messages": []}
            MockPublisher.return_value = AsyncMock()
            mock_extract.kiq = AsyncMock()
            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        # First save should set RUNNING
        first_save_call = mock_chat_session.save.call_args_list[0]
        assert "current_execution_status" in first_save_call.kwargs.get("update_fields", [])

    @pytest.mark.asyncio
    async def test_sets_status_to_completed_on_success(self, mock_user, mock_workflow, mock_chat_session):
        """Test that session status is COMPLETED after successful execution."""
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.extract_session_memories") as mock_extract, \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            mock_extract.kiq = AsyncMock()
            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_stream.return_value = {"messages": []}
            MockPublisher.return_value = AsyncMock()
            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        assert mock_chat_session.current_execution_status == ChatExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_sets_status_to_failed_on_cost_cap_exceeded(self, mock_user, mock_workflow, mock_chat_session):
        """Test that session is FAILED with cost_cap_exceeded error."""
        from seer.observability.exceptions import RunCostCapExceeded
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"):

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            MockPublisher.return_value = AsyncMock()
            mock_stream.side_effect = RunCostCapExceeded(
                run_identifier="thread-123",
                accumulated_cost=10.0,
                cost_cap=5.0,
                run_type="chat"
            )

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        assert mock_chat_session.current_execution_status == ChatExecutionStatus.FAILED
        assert mock_chat_session.current_execution_error["type"] == "cost_cap_exceeded"
        assert mock_chat_session.current_execution_error["status"] == 402

    @pytest.mark.asyncio
    async def test_sets_status_to_failed_on_exception(self, mock_user, mock_workflow, mock_chat_session):
        """Test that session is FAILED with execution_error on exception."""
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock) as mock_checkpointer:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(side_effect=RuntimeError("Database error"))

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        assert mock_chat_session.current_execution_status == ChatExecutionStatus.FAILED
        assert mock_chat_session.current_execution_error["type"] == "execution_error"
        assert mock_chat_session.current_execution_error["status"] == 500

    @pytest.mark.asyncio
    async def test_sets_status_to_interrupted_when_interrupt_detected(self, mock_user, mock_workflow, mock_chat_session):
        """Test that session is INTERRUPTED when agent needs user input."""
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"):

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_stream.return_value = {"messages": [MagicMock(content="Need approval")]}
            MockPublisher.return_value = AsyncMock()
            MockInterrupt.extract_interrupt_from_result.return_value = (True, {"type": "approval_request"})
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            mock_thinking.return_value = []

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        assert mock_chat_session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        assert mock_chat_session.pending_interrupt_type == "approval_request"

    @pytest.mark.asyncio
    async def test_saves_assistant_message_on_completion(self, mock_user, mock_workflow, mock_chat_session):
        """Test that assistant message is saved on successful completion."""
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock) as mock_save_msg, \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())

            last_msg = MagicMock()
            last_msg.content = "I can help with that!"
            mock_stream.return_value = {"messages": [last_msg]}
            MockPublisher.return_value = AsyncMock()
            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            mock_thinking.return_value = ["thinking step 1"]
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        mock_save_msg.assert_called_once()
        call_kwargs = mock_save_msg.call_args.kwargs
        assert call_kwargs["session_id"] == 1
        assert call_kwargs["role"] == "assistant"
        assert call_kwargs["content"] == "I can help with that!"

    @pytest.mark.asyncio
    async def test_includes_proposal_in_message(self, mock_user, mock_workflow, mock_chat_session):
        """Test that proposal spec is included in saved message."""
        from seer.worker.tasks.chat import chat_execution_task

        mock_proposal = MagicMock()
        mock_proposal.spec = {"version": "2", "nodes": [{"id": "new_node"}]}

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock) as mock_save_msg, \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_stream.return_value = {"messages": []}
            MockPublisher.return_value = AsyncMock()
            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            mock_thinking.return_value = []

            # Mock the chained get_or_none().prefetch_related()
            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_proposal)
            MockProposal.get_or_none.return_value = mock_query

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        call_kwargs = mock_save_msg.call_args.kwargs
        assert call_kwargs["suggested_edits"] == mock_proposal.spec
        assert call_kwargs["proposal"] == mock_proposal

    @pytest.mark.asyncio
    async def test_clears_runtime_context_on_success(self, mock_user, mock_workflow, mock_chat_session):
        """Test that runtime context is cleared after successful execution."""
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context") as mock_clear, \
             patch("seer.worker.tasks.chat.extract_session_memories") as mock_extract, \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_stream.return_value = {"messages": []}
            MockPublisher.return_value = AsyncMock()
            mock_extract.kiq = AsyncMock()
            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_clears_runtime_context_on_cost_cap_error(self, mock_user, mock_workflow, mock_chat_session):
        """Test that runtime context is cleared even on cost cap error."""
        from seer.observability.exceptions import RunCostCapExceeded
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context") as mock_clear:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            MockPublisher.return_value = AsyncMock()
            mock_stream.side_effect = RunCostCapExceeded(
                run_identifier="t", accumulated_cost=10.0, cost_cap=5.0, run_type="chat"
            )

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_sets_sentry_context(self, mock_user, mock_workflow, mock_chat_session):
        """Test that Sentry context is set before execution."""
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.extract_session_memories") as mock_extract, \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal, \
             patch("seer.worker.tasks.chat._set_sentry_context_for_chat") as mock_sentry:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_stream.return_value = {"messages": []}
            MockPublisher.return_value = AsyncMock()
            mock_extract.kiq = AsyncMock()
            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
            )

        mock_sentry.assert_called_once_with(mock_user, 1, 1, mock_chat_session.thread_id)

    @pytest.mark.asyncio
    async def test_uses_model_override_when_provided(self, mock_user, mock_workflow, mock_chat_session):
        """Test that model parameter overrides default."""
        from seer.worker.tasks.chat import chat_execution_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent") as mock_create_agent, \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.extract_session_memories") as mock_extract, \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_stream.return_value = {"messages": []}
            MockPublisher.return_value = AsyncMock()
            mock_extract.kiq = AsyncMock()
            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Hello",
                workflow_id=1,
                model="gpt-4-turbo",
            )

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_state_fallback_detects_interrupt_when_result_has_none(self, mock_user, mock_workflow, mock_chat_session):
        """Test that interrupt is detected via agent state when astream_events doesn't surface __interrupt__."""
        from seer.worker.tasks.chat import chat_execution_task

        interrupt_data = {"type": "clarification_questions", "questions": ["What is the goal?"]}

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent"), \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat._stream_agent_with_orchestrator", new_callable=AsyncMock) as mock_stream, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"):

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_stream.return_value = {"messages": [MagicMock(content="Let me ask some questions")]}
            mock_thinking.return_value = []

            mock_publisher_instance = AsyncMock()
            MockPublisher.return_value = mock_publisher_instance

            # from_result finds nothing; from_state finds the interrupt in the checkpoint
            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(True, interrupt_data))

            await chat_execution_task(
                session_id=1,
                user_id=1,
                message="Help me build a workflow",
                workflow_id=1,
            )

        assert mock_chat_session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        assert mock_chat_session.pending_interrupt_type == "clarification_questions"
        MockInterrupt.extract_interrupt_from_state.assert_awaited_once()


# =============================================================================
# Chat Resume Task Tests
# =============================================================================


@pytest.mark.unit
class TestChatResumeTask:
    """Tests for chat_resume_task function."""

    @pytest.mark.asyncio
    async def test_clears_interrupt_state_on_start(self, mock_user, mock_workflow, mock_chat_session):
        """Test that interrupt state is cleared when resume starts."""
        from seer.worker.tasks.chat import chat_resume_task

        mock_chat_session.pending_interrupt_type = "approval_request"
        mock_chat_session.pending_interrupt_data = {"type": "approval_request"}

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent") as mock_create, \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.merge_nexus_langfuse_callbacks") as mock_merge, \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat._current_thread_id") as mock_thread_var, \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_merge.return_value = {}
            mock_thinking.return_value = []
            MockPublisher.return_value = AsyncMock()

            mock_agent = MagicMock()
            mock_agent.astream_events = _empty_astream_events
            mock_create.return_value = mock_agent

            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_resume_task(
                session_id=1,
                user_id=1,
                thread_id="thread-123",
                resume_command_data={"approved": True},
                workflow_id=1,
            )

        assert mock_chat_session.pending_interrupt_type is None
        assert mock_chat_session.pending_interrupt_data is None

    @pytest.mark.asyncio
    async def test_sets_thread_id_context_variable(self, mock_user, mock_workflow, mock_chat_session):
        """Test that _current_thread_id context variable is set."""
        from seer.worker.tasks.chat import chat_resume_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent") as mock_create, \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.merge_nexus_langfuse_callbacks") as mock_merge, \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat._current_thread_id") as mock_thread_var, \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_merge.return_value = {}
            mock_thinking.return_value = []
            MockPublisher.return_value = AsyncMock()

            mock_agent = MagicMock()
            mock_agent.astream_events = _empty_astream_events
            mock_create.return_value = mock_agent

            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_resume_task(
                session_id=1,
                user_id=1,
                thread_id="thread-xyz",
                resume_command_data={},
                workflow_id=1,
            )

        mock_thread_var.set.assert_called_with("thread-xyz")

    @pytest.mark.asyncio
    async def test_resets_thread_id_on_completion(self, mock_user, mock_workflow, mock_chat_session):
        """Test that thread_id context is reset in finally block."""
        from seer.worker.tasks.chat import chat_resume_task

        mock_token = MagicMock()

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent") as mock_create, \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.merge_nexus_langfuse_callbacks") as mock_merge, \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat._current_thread_id") as mock_thread_var, \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_merge.return_value = {}
            mock_thinking.return_value = []
            mock_thread_var.set.return_value = mock_token
            MockPublisher.return_value = AsyncMock()

            mock_agent = MagicMock()
            mock_agent.astream_events = _empty_astream_events
            mock_create.return_value = mock_agent

            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_resume_task(
                session_id=1,
                user_id=1,
                thread_id="thread-123",
                resume_command_data={},
                workflow_id=1,
            )

        mock_thread_var.reset.assert_called_with(mock_token)

    @pytest.mark.asyncio
    async def test_handles_another_interrupt(self, mock_user, mock_workflow, mock_chat_session):
        """Test that resume can result in another interrupt."""
        from seer.worker.tasks.chat import chat_resume_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent") as mock_create, \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.merge_nexus_langfuse_callbacks") as mock_merge, \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat._current_thread_id"):

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_merge.return_value = {}
            mock_thinking.return_value = []
            MockPublisher.return_value = AsyncMock()

            mock_agent = MagicMock()
            mock_agent.astream_events = _empty_astream_events
            mock_create.return_value = mock_agent

            MockInterrupt.extract_interrupt_from_result.return_value = (True, {"type": "second_approval"})

            await chat_resume_task(
                session_id=1,
                user_id=1,
                thread_id="thread-123",
                resume_command_data={},
                workflow_id=1,
            )

        assert mock_chat_session.current_execution_status == ChatExecutionStatus.INTERRUPTED
        assert mock_chat_session.pending_interrupt_type == "second_approval"

    @pytest.mark.asyncio
    async def test_sets_status_to_completed_on_success(self, mock_user, mock_workflow, mock_chat_session):
        """Test that session is COMPLETED after successful resume."""
        from seer.worker.tasks.chat import chat_resume_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow, \
             patch("seer.worker.tasks.chat.get_checkpointer", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.create_nexus_chat_agent") as mock_create, \
             patch("seer.worker.tasks.chat._get_user_settings_and_context", new_callable=AsyncMock) as mock_settings, \
             patch("seer.worker.tasks.chat.StreamPublisher") as MockPublisher, \
             patch("seer.worker.tasks.chat.InterruptHandler") as MockInterrupt, \
             patch("seer.worker.tasks.chat.save_chat_message", new_callable=AsyncMock), \
             patch("seer.worker.tasks.chat.extract_thinking_from_messages") as mock_thinking, \
             patch("seer.worker.tasks.chat.langfuse_user_context", mock_langfuse_context), \
             patch("seer.worker.tasks.chat.merge_nexus_langfuse_callbacks") as mock_merge, \
             patch("seer.worker.tasks.chat.set_chat_runtime_context"), \
             patch("seer.worker.tasks.chat.clear_chat_runtime_context"), \
             patch("seer.worker.tasks.chat._current_thread_id"), \
             patch("seer.worker.tasks.chat.WorkflowProposal") as MockProposal:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(return_value=mock_workflow)
            mock_settings.return_value = (50, MagicMock())
            mock_merge.return_value = {}
            mock_thinking.return_value = []
            MockPublisher.return_value = AsyncMock()

            mock_agent = MagicMock()
            mock_agent.astream_events = _empty_astream_events
            mock_create.return_value = mock_agent

            MockInterrupt.extract_interrupt_from_result.return_value = (False, None)
            MockInterrupt.extract_interrupt_from_state = AsyncMock(return_value=(False, None))
            MockProposal.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            await chat_resume_task(
                session_id=1,
                user_id=1,
                thread_id="thread-123",
                resume_command_data={},
                workflow_id=1,
            )

        assert mock_chat_session.current_execution_status == ChatExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_sets_status_to_failed_on_exception(self, mock_user, mock_workflow, mock_chat_session):
        """Test that session is FAILED on exception during resume."""
        from seer.worker.tasks.chat import chat_resume_task

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.User") as MockUser, \
             patch("seer.worker.tasks.chat.Workflow") as MockWorkflow:

            MockSession.get = AsyncMock(return_value=mock_chat_session)
            MockUser.get = AsyncMock(return_value=mock_user)
            MockWorkflow.get = AsyncMock(side_effect=RuntimeError("DB error"))

            await chat_resume_task(
                session_id=1,
                user_id=1,
                thread_id="thread-123",
                resume_command_data={},
                workflow_id=1,
            )

        assert mock_chat_session.current_execution_status == ChatExecutionStatus.FAILED
        assert mock_chat_session.current_execution_error["type"] == "execution_error"


# =============================================================================
# Cleanup Stale Chat Executions Tests
# =============================================================================


@pytest.mark.unit
class TestCleanupStaleChatExecutions:
    """Tests for cleanup_stale_chat_executions function."""

    @pytest.mark.asyncio
    async def test_finds_stale_queued_sessions(self):
        """Test that QUEUED sessions older than 1 hour are found."""
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        stale_session = MagicMock()
        stale_session.current_execution_status = ChatExecutionStatus.QUEUED
        stale_session.save = AsyncMock()

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession:
            mock_filter = MagicMock()
            mock_filter.all = AsyncMock(return_value=[stale_session])
            MockSession.filter.return_value = mock_filter

            await cleanup_stale_chat_executions()

        assert stale_session.current_execution_status == ChatExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_finds_stale_running_sessions(self):
        """Test that RUNNING sessions older than 1 hour are found."""
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        stale_session = MagicMock()
        stale_session.current_execution_status = ChatExecutionStatus.RUNNING
        stale_session.save = AsyncMock()

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession:
            mock_filter = MagicMock()
            mock_filter.all = AsyncMock(return_value=[stale_session])
            MockSession.filter.return_value = mock_filter

            await cleanup_stale_chat_executions()

        assert stale_session.current_execution_status == ChatExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_sets_timeout_error(self):
        """Test that cleaned sessions have correct timeout error."""
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        stale_session = MagicMock()
        stale_session.save = AsyncMock()

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession:
            mock_filter = MagicMock()
            mock_filter.all = AsyncMock(return_value=[stale_session])
            MockSession.filter.return_value = mock_filter

            await cleanup_stale_chat_executions()

        assert stale_session.current_execution_error["type"] == "timeout"
        assert stale_session.current_execution_error["reason"] == "cleanup_task"
        assert stale_session.current_execution_error["status"] == 500

    @pytest.mark.asyncio
    async def test_sets_finished_at_timestamp(self):
        """Test that finished_at is set on cleaned sessions."""
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        stale_session = MagicMock()
        stale_session.save = AsyncMock()

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession:
            mock_filter = MagicMock()
            mock_filter.all = AsyncMock(return_value=[stale_session])
            MockSession.filter.return_value = mock_filter

            await cleanup_stale_chat_executions()

        assert stale_session.current_execution_finished_at is not None

    @pytest.mark.asyncio
    async def test_handles_no_stale_sessions(self):
        """Test handling when no stale sessions exist."""
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.logger") as mock_logger:
            mock_filter = MagicMock()
            mock_filter.all = AsyncMock(return_value=[])
            MockSession.filter.return_value = mock_filter

            await cleanup_stale_chat_executions()

        # Should log 0 cleaned
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args
        assert "0" in str(call_args)

    @pytest.mark.asyncio
    async def test_logs_cleanup_count(self):
        """Test that cleanup count is logged."""
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        sessions = [MagicMock(save=AsyncMock()) for _ in range(3)]

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession, \
             patch("seer.worker.tasks.chat.logger") as mock_logger:
            mock_filter = MagicMock()
            mock_filter.all = AsyncMock(return_value=sessions)
            MockSession.filter.return_value = mock_filter

            await cleanup_stale_chat_executions()

        mock_logger.info.assert_called_with("Cleaned up %d stale chat executions", 3)

    @pytest.mark.asyncio
    async def test_uses_correct_filter_criteria(self):
        """Test that filter uses correct status and time criteria."""
        from seer.worker.tasks.chat import cleanup_stale_chat_executions

        with patch("seer.worker.tasks.chat.WorkflowChatSession") as MockSession:
            mock_filter = MagicMock()
            mock_filter.all = AsyncMock(return_value=[])
            MockSession.filter.return_value = mock_filter

            await cleanup_stale_chat_executions()

        call_kwargs = MockSession.filter.call_args.kwargs
        assert "current_execution_status__in" in call_kwargs
        assert ChatExecutionStatus.QUEUED in call_kwargs["current_execution_status__in"]
        assert ChatExecutionStatus.RUNNING in call_kwargs["current_execution_status__in"]
        assert "current_execution_started_at__lt" in call_kwargs
