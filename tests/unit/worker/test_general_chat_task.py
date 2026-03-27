"""
Unit tests for worker.tasks.general_chat module.

Tests:
- Message format conversion
- Status transitions on success and failure
- Image generation failure isolation
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.database.workflow_models import ChatExecutionStatus


# =============================================================================
# Message Conversion
# =============================================================================


@pytest.mark.unit
class TestBuildLangchainMessages:
    """Tests for _build_langchain_messages."""

    def test_user_message_converts_to_human(self):
        from seer.worker.tasks.general_chat import _build_langchain_messages

        msg = MagicMock()
        msg.role = "user"
        msg.content = "Hello"

        result = _build_langchain_messages([msg])
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_assistant_message_converts_to_ai(self):
        from seer.worker.tasks.general_chat import _build_langchain_messages

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "Hi there"

        result = _build_langchain_messages([msg])
        assert len(result) == 1
        assert result[0].content == "Hi there"

    def test_unknown_role_skipped(self):
        from seer.worker.tasks.general_chat import _build_langchain_messages

        msg = MagicMock()
        msg.role = "system"
        msg.content = "System message"

        result = _build_langchain_messages([msg])
        assert len(result) == 0

    def test_mixed_messages(self):
        from seer.worker.tasks.general_chat import _build_langchain_messages

        user_msg = MagicMock(role="user", content="Question")
        assistant_msg = MagicMock(role="assistant", content="Answer")

        result = _build_langchain_messages([user_msg, assistant_msg])
        assert len(result) == 2

    def test_empty_list(self):
        from seer.worker.tasks.general_chat import _build_langchain_messages

        result = _build_langchain_messages([])
        assert result == []


# =============================================================================
# General Chat Task
# =============================================================================


@pytest.mark.unit
class TestGeneralChatTask:
    """Tests for general_chat_task."""

    @pytest.mark.asyncio
    async def test_success_sets_completed_status(self):
        from seer.worker.tasks.general_chat import general_chat_task

        session = MagicMock()
        session.save = AsyncMock()

        llm_result = MagicMock()
        llm_result.content = "Here's the answer"

        with (
            patch("seer.worker.tasks.general_chat.GeneralChatSession") as mock_session_model,
            patch("seer.worker.tasks.general_chat.GeneralChatMessage") as mock_msg_model,
            patch("seer.worker.tasks.general_chat.get_llm") as mock_get_llm,
            patch("seer.api.chat.services.save_message", new_callable=AsyncMock),
        ):
            mock_session_model.get = AsyncMock(return_value=session)
            mock_msg_qs = MagicMock()
            mock_msg_qs.order_by.return_value.all = AsyncMock(return_value=[])
            mock_msg_model.filter.return_value = mock_msg_qs

            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = llm_result
            mock_get_llm.return_value = mock_llm

            await general_chat_task(session_id=1, message="test", user_id=1)

        assert session.current_execution_status == ChatExecutionStatus.COMPLETED
        assert session.current_execution_error is None

    @pytest.mark.asyncio
    async def test_failure_sets_failed_status(self):
        from seer.worker.tasks.general_chat import general_chat_task

        session = MagicMock()
        session.save = AsyncMock()

        with (
            patch("seer.worker.tasks.general_chat.GeneralChatSession") as mock_session_model,
            patch("seer.worker.tasks.general_chat.GeneralChatMessage") as mock_msg_model,
            patch("seer.worker.tasks.general_chat.get_llm") as mock_get_llm,
        ):
            mock_session_model.get = AsyncMock(return_value=session)
            mock_msg_qs = MagicMock()
            mock_msg_qs.order_by.return_value.all = AsyncMock(return_value=[])
            mock_msg_model.filter.return_value = mock_msg_qs

            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = RuntimeError("LLM is down")
            mock_get_llm.return_value = mock_llm

            await general_chat_task(session_id=1, message="test", user_id=1)

        assert session.current_execution_status == ChatExecutionStatus.FAILED
        assert session.current_execution_error["type"] == "execution_error"
        assert session.current_execution_error["status"] == 500

    @pytest.mark.asyncio
    async def test_image_generation_failure_does_not_fail_chat(self):
        """Image gen failing should not prevent the chat from completing."""
        from seer.worker.tasks.general_chat import general_chat_task

        session = MagicMock()
        session.save = AsyncMock()

        llm_result = MagicMock()
        llm_result.content = "Chat response"

        with (
            patch("seer.worker.tasks.general_chat.GeneralChatSession") as mock_session_model,
            patch("seer.worker.tasks.general_chat.GeneralChatMessage") as mock_msg_model,
            patch("seer.worker.tasks.general_chat.get_llm") as mock_get_llm,
            patch("seer.api.chat.services.save_message", new_callable=AsyncMock),
            patch("seer.services.image_gen.generate_image", side_effect=RuntimeError("Image gen down")),
        ):
            mock_session_model.get = AsyncMock(return_value=session)
            mock_msg_qs = MagicMock()
            mock_msg_qs.order_by.return_value.all = AsyncMock(return_value=[])
            mock_msg_model.filter.return_value = mock_msg_qs

            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = llm_result
            mock_get_llm.return_value = mock_llm

            await general_chat_task(
                session_id=1, message="test", user_id=1, generate_image=True,
            )

        # Chat should still complete despite image generation failure
        assert session.current_execution_status == ChatExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_sets_running_status_before_execution(self):
        from seer.worker.tasks.general_chat import general_chat_task

        session = MagicMock()
        session.save = AsyncMock()
        statuses_seen = []

        original_save = session.save.side_effect

        async def track_save(*args, **kwargs):
            statuses_seen.append(session.current_execution_status)

        session.save.side_effect = track_save

        llm_result = MagicMock()
        llm_result.content = "Done"

        with (
            patch("seer.worker.tasks.general_chat.GeneralChatSession") as mock_session_model,
            patch("seer.worker.tasks.general_chat.GeneralChatMessage") as mock_msg_model,
            patch("seer.worker.tasks.general_chat.get_llm") as mock_get_llm,
            patch("seer.api.chat.services.save_message", new_callable=AsyncMock),
        ):
            mock_session_model.get = AsyncMock(return_value=session)
            mock_msg_qs = MagicMock()
            mock_msg_qs.order_by.return_value.all = AsyncMock(return_value=[])
            mock_msg_model.filter.return_value = mock_msg_qs

            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = llm_result
            mock_get_llm.return_value = mock_llm

            await general_chat_task(session_id=1, message="test", user_id=1)

        # First save should be RUNNING, second should be COMPLETED
        assert statuses_seen[0] == ChatExecutionStatus.RUNNING
        assert statuses_seen[-1] == ChatExecutionStatus.COMPLETED
