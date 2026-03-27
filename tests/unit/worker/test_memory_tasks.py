"""
Unit tests for worker.tasks.memory module.

Tests:
- Message formatting for extraction
- Config gating (memory_enabled flags)
- Extraction success/failure handling
- Backfill logic
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Message Formatting
# =============================================================================


@pytest.mark.unit
class TestFormatMessagesForExtraction:
    """Tests for _format_messages_for_extraction."""

    def test_user_and_assistant_messages(self):
        from seer.worker.tasks.memory import _format_messages_for_extraction

        msg1 = MagicMock(role="user", content="Hello")
        msg2 = MagicMock(role="assistant", content="Hi there!")

        result = _format_messages_for_extraction([msg1, msg2])
        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result

    def test_truncates_long_messages(self):
        from seer.worker.tasks.memory import _format_messages_for_extraction

        msg = MagicMock(role="user", content="x" * 3000)
        result = _format_messages_for_extraction([msg])
        assert len(result) < 3000
        assert result.endswith("...")

    def test_empty_messages(self):
        from seer.worker.tasks.memory import _format_messages_for_extraction

        result = _format_messages_for_extraction([])
        assert result == ""

    def test_messages_separated_by_double_newline(self):
        from seer.worker.tasks.memory import _format_messages_for_extraction

        msg1 = MagicMock(role="user", content="Q")
        msg2 = MagicMock(role="assistant", content="A")
        result = _format_messages_for_extraction([msg1, msg2])
        assert "\n\n" in result


# =============================================================================
# Extract Session Memories
# =============================================================================


@pytest.mark.unit
class TestExtractSessionMemories:
    """Tests for extract_session_memories task."""

    @pytest.mark.asyncio
    async def test_skips_when_memory_disabled(self):
        from seer.worker.tasks.memory import extract_session_memories

        with patch("seer.worker.tasks.memory.config") as mock_config:
            mock_config.memory_enabled = False
            mock_config.memory_extraction_enabled = True

            result = await extract_session_memories(session_id=1)

        assert result["skipped"] is True
        assert "disabled" in result["reason"]

    @pytest.mark.asyncio
    async def test_skips_when_extraction_disabled(self):
        from seer.worker.tasks.memory import extract_session_memories

        with patch("seer.worker.tasks.memory.config") as mock_config:
            mock_config.memory_enabled = True
            mock_config.memory_extraction_enabled = False

            result = await extract_session_memories(session_id=1)

        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_skips_when_no_messages(self):
        from seer.worker.tasks.memory import extract_session_memories

        session = MagicMock()
        session.user = MagicMock(user_id="user_123")
        session.workflow = None

        with (
            patch("seer.worker.tasks.memory.config") as mock_config,
            patch("seer.worker.tasks.memory.WorkflowChatSession") as mock_session_model,
            patch("seer.worker.tasks.memory.WorkflowChatMessage") as mock_msg_model,
        ):
            mock_config.memory_enabled = True
            mock_config.memory_extraction_enabled = True
            mock_session_model.get.return_value.prefetch_related = AsyncMock(return_value=session)
            mock_msg_qs = MagicMock()
            mock_msg_qs.order_by.return_value.all = AsyncMock(return_value=[])
            mock_msg_model.filter.return_value = mock_msg_qs

            result = await extract_session_memories(session_id=1)

        assert result["skipped"] is True
        assert result["reason"] == "no_messages"

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        from seer.worker.tasks.memory import extract_session_memories

        session = MagicMock()
        session.user = MagicMock(user_id="user_123")
        session.workflow = MagicMock(workflow_id=None)
        session.workflow_id = None
        session.thread_id = "thread-1"
        session.title = "Test Session"

        msg = MagicMock(role="user", content="Remember my name is John")

        mock_mem_service = AsyncMock()
        mock_mem_service.add_memory.return_value = {"results": [{"id": "m1"}, {"id": "m2"}]}

        with (
            patch("seer.worker.tasks.memory.config") as mock_config,
            patch("seer.worker.tasks.memory.WorkflowChatSession") as mock_session_model,
            patch("seer.worker.tasks.memory.WorkflowChatMessage") as mock_msg_model,
            patch("seer.worker.tasks.memory.UserMemoryService", return_value=mock_mem_service),
        ):
            mock_config.memory_enabled = True
            mock_config.memory_extraction_enabled = True
            mock_session_model.get.return_value.prefetch_related = AsyncMock(return_value=session)
            mock_msg_qs = MagicMock()
            mock_msg_qs.order_by.return_value.all = AsyncMock(return_value=[msg])
            mock_msg_model.filter.return_value = mock_msg_qs

            result = await extract_session_memories(session_id=1)

        assert result["success"] is True
        assert result["memories_extracted"] == 2
        assert result["user_id"] == "user_123"

    @pytest.mark.asyncio
    async def test_extraction_failure_returns_error_dict(self):
        """Memory extraction failure should NOT crash the worker."""
        from seer.worker.tasks.memory import extract_session_memories

        with (
            patch("seer.worker.tasks.memory.config") as mock_config,
            patch("seer.worker.tasks.memory.WorkflowChatSession") as mock_session_model,
        ):
            mock_config.memory_enabled = True
            mock_config.memory_extraction_enabled = True
            mock_session_model.get.return_value.prefetch_related = AsyncMock(
                side_effect=RuntimeError("DB connection lost")
            )

            result = await extract_session_memories(session_id=1)

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_mem0_returns_none(self):
        from seer.worker.tasks.memory import extract_session_memories

        session = MagicMock()
        session.user = MagicMock(user_id="user_123")
        session.workflow = None
        session.workflow_id = None
        session.thread_id = "thread-1"
        session.title = None

        msg = MagicMock(role="user", content="test")

        mock_mem_service = AsyncMock()
        mock_mem_service.add_memory.return_value = None

        with (
            patch("seer.worker.tasks.memory.config") as mock_config,
            patch("seer.worker.tasks.memory.WorkflowChatSession") as mock_session_model,
            patch("seer.worker.tasks.memory.WorkflowChatMessage") as mock_msg_model,
            patch("seer.worker.tasks.memory.UserMemoryService", return_value=mock_mem_service),
        ):
            mock_config.memory_enabled = True
            mock_config.memory_extraction_enabled = True
            mock_session_model.get.return_value.prefetch_related = AsyncMock(return_value=session)
            mock_msg_qs = MagicMock()
            mock_msg_qs.order_by.return_value.all = AsyncMock(return_value=[msg])
            mock_msg_model.filter.return_value = mock_msg_qs

            result = await extract_session_memories(session_id=1)

        assert result["success"] is False
        assert result["reason"] == "extraction_failed"


# =============================================================================
# Backfill User Memories
# =============================================================================


@pytest.mark.unit
class TestBackfillUserMemories:
    """Tests for backfill_user_memories task."""

    @pytest.mark.asyncio
    async def test_skips_when_memory_disabled(self):
        from seer.worker.tasks.memory import backfill_user_memories

        with patch("seer.worker.tasks.memory.config") as mock_config:
            mock_config.memory_enabled = False

            result = await backfill_user_memories(user_id_internal=1)

        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_backfill_processes_sessions(self):
        from seer.worker.tasks.memory import backfill_user_memories

        session1 = MagicMock(id=10)
        session2 = MagicMock(id=20)

        with (
            patch("seer.worker.tasks.memory.config") as mock_config,
            patch("seer.worker.tasks.memory.WorkflowChatSession") as mock_session_model,
            patch("seer.worker.tasks.memory.extract_session_memories", new_callable=AsyncMock) as mock_extract,
        ):
            mock_config.memory_enabled = True
            mock_config.memory_extraction_enabled = True
            mock_qs = MagicMock()
            mock_qs.order_by.return_value.limit.return_value.all = AsyncMock(
                return_value=[session1, session2]
            )
            mock_session_model.filter.return_value = mock_qs
            mock_extract.side_effect = [
                {"success": True, "memories_extracted": 3},
                {"success": True, "memories_extracted": 1},
            ]

            result = await backfill_user_memories(user_id_internal=1, limit=50)

        assert result["success"] is True
        assert result["sessions_processed"] == 2
        assert result["sessions_successful"] == 2
        assert result["total_memories_extracted"] == 4

    @pytest.mark.asyncio
    async def test_backfill_handles_partial_failures(self):
        from seer.worker.tasks.memory import backfill_user_memories

        session1 = MagicMock(id=10)
        session2 = MagicMock(id=20)

        with (
            patch("seer.worker.tasks.memory.config") as mock_config,
            patch("seer.worker.tasks.memory.WorkflowChatSession") as mock_session_model,
            patch("seer.worker.tasks.memory.extract_session_memories", new_callable=AsyncMock) as mock_extract,
        ):
            mock_config.memory_enabled = True
            mock_config.memory_extraction_enabled = True
            mock_qs = MagicMock()
            mock_qs.order_by.return_value.limit.return_value.all = AsyncMock(
                return_value=[session1, session2]
            )
            mock_session_model.filter.return_value = mock_qs
            mock_extract.side_effect = [
                {"success": True, "memories_extracted": 2},
                {"success": False, "error": "Mem0 timeout"},
            ]

            result = await backfill_user_memories(user_id_internal=1)

        assert result["success"] is True
        assert result["sessions_successful"] == 1
        assert result["total_memories_extracted"] == 2
