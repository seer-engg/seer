"""
Unit tests for UserMemoryService.

Tests the memory service layer with mocked Mem0 client to verify:
- Memory add/search/get operations
- Context formatting for prompt injection
- Graceful degradation when service unavailable
- Filter application
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from seer.services.memory.user_memory import UserMemoryService, get_user_memory_service


@pytest.fixture
def mock_mem0_client():
    """Create a mock Mem0 client."""
    client = MagicMock()
    client.add = MagicMock(return_value={
        "results": [
            {"id": "mem_1", "memory": "User prefers Slack notifications"},
            {"id": "mem_2", "memory": "User's company uses PostgreSQL"},
        ]
    })
    client.search = MagicMock(return_value={
        "results": [
            {"id": "mem_1", "memory": "User prefers Slack notifications", "score": 0.95},
            {"id": "mem_2", "memory": "User uses Gmail for work", "score": 0.82},
        ]
    })
    client.get_all = MagicMock(return_value={
        "results": [
            {"id": "mem_1", "memory": "User prefers Slack"},
            {"id": "mem_2", "memory": "User uses PostgreSQL"},
            {"id": "mem_3", "memory": "User likes dark mode"},
        ]
    })
    client.delete = MagicMock()
    return client


@pytest.fixture
def memory_service(mock_mem0_client):
    """Create a UserMemoryService with mocked client."""
    service = UserMemoryService()
    service._client = mock_mem0_client
    return service


class TestUserMemoryServiceBasics:
    """Test basic service operations."""

    @pytest.mark.unit
    async def test_is_available_when_enabled(self, memory_service):
        """Service should report available when enabled with client."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True
            assert memory_service.is_available is True

    @pytest.mark.unit
    async def test_is_available_when_disabled(self, memory_service):
        """Service should report unavailable when disabled."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = False
            assert memory_service.is_available is False

    @pytest.mark.unit
    async def test_is_available_when_no_client(self):
        """Service should report unavailable when client is None."""
        service = UserMemoryService()
        service._client = None
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True
            with patch("seer.services.memory.user_memory.get_mem0_client", return_value=None):
                assert service.is_available is False


class TestAddMemory:
    """Test memory addition."""

    @pytest.mark.unit
    async def test_add_memory_success(self, memory_service, mock_mem0_client):
        """Should add memory with metadata."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True

            result = await memory_service.add_memory(
                user_id="user_123",
                content="I prefer using Python for automation",
                metadata={"session_id": 42},
            )

            assert result is not None
            assert "results" in result
            mock_mem0_client.add.assert_called_once()

            # Verify call arguments
            call_args = mock_mem0_client.add.call_args
            assert call_args[0][0] == "I prefer using Python for automation"
            assert call_args[1]["user_id"] == "user_123"
            assert "added_at" in call_args[1]["metadata"]
            assert call_args[1]["metadata"]["session_id"] == 42

    @pytest.mark.unit
    async def test_add_memory_when_disabled(self, memory_service):
        """Should return None when memory disabled."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = False

            result = await memory_service.add_memory(
                user_id="user_123",
                content="Test content",
            )

            assert result is None

    @pytest.mark.unit
    async def test_add_memory_handles_exception(self, memory_service, mock_mem0_client):
        """Should handle exceptions gracefully."""
        mock_mem0_client.add.side_effect = Exception("Connection failed")

        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True

            result = await memory_service.add_memory(
                user_id="user_123",
                content="Test content",
            )

            # Should not raise, should return None
            assert result is None


class TestSearchMemory:
    """Test memory search."""

    @pytest.mark.unit
    async def test_search_success(self, memory_service, mock_mem0_client):
        """Should search memories and return results."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True

            results = await memory_service.search(
                user_id="user_123",
                query="notification preferences",
                limit=5,
            )

            assert len(results) == 2
            assert results[0]["memory"] == "User prefers Slack notifications"
            mock_mem0_client.search.assert_called_once()

    @pytest.mark.unit
    async def test_search_when_disabled(self, memory_service):
        """Should return empty list when disabled."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = False

            results = await memory_service.search(
                user_id="user_123",
                query="test query",
            )

            assert results == []

    @pytest.mark.unit
    async def test_search_with_filters(self, memory_service, mock_mem0_client):
        """Should apply metadata filters to results."""
        # Add session_id to some results
        mock_mem0_client.search.return_value = {
            "results": [
                {"id": "1", "memory": "Memory 1", "metadata": {"session_id": 1}},
                {"id": "2", "memory": "Memory 2", "metadata": {}},
                {"id": "3", "memory": "Memory 3", "metadata": {"session_id": 2}},
            ]
        }

        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True

            results = await memory_service.search(
                user_id="user_123",
                query="test",
                filters={"has_session_id": True},
            )

            # Should only return memories with session_id
            assert len(results) == 2
            assert all(m.get("metadata", {}).get("session_id") for m in results)


class TestGetAll:
    """Test getting all memories."""

    @pytest.mark.unit
    async def test_get_all_success(self, memory_service, mock_mem0_client):
        """Should get all memories for user."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True

            results = await memory_service.get_all(user_id="user_123")

            assert len(results) == 3
            mock_mem0_client.get_all.assert_called_once_with(user_id="user_123")

    @pytest.mark.unit
    async def test_get_all_when_disabled(self, memory_service):
        """Should return empty list when disabled."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = False

            results = await memory_service.get_all(user_id="user_123")

            assert results == []


class TestDeleteMemory:
    """Test memory deletion."""

    @pytest.mark.unit
    async def test_delete_memory_success(self, memory_service, mock_mem0_client):
        """Should delete memory by ID."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True

            result = await memory_service.delete_memory("mem_123")

            assert result is True
            mock_mem0_client.delete.assert_called_once_with("mem_123")

    @pytest.mark.unit
    async def test_delete_memory_when_disabled(self, memory_service):
        """Should return False when disabled."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = False

            result = await memory_service.delete_memory("mem_123")

            assert result is False


class TestContextForPrompt:
    """Test context formatting for prompt injection."""

    @pytest.mark.unit
    async def test_get_context_with_query(self, memory_service, mock_mem0_client):
        """Should search with query and format results."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True
            mock_config.memory_context_injection_enabled = True
            mock_config.memory_context_max_memories = 10

            context = await memory_service.get_context_for_prompt(
                user_id="user_123",
                current_query="How do I set up notifications?",
            )

            assert "## User Context" in context
            assert "User prefers Slack notifications" in context
            mock_mem0_client.search.assert_called_once()

    @pytest.mark.unit
    async def test_get_context_without_query(self, memory_service, mock_mem0_client):
        """Should get recent memories when no query provided."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True
            mock_config.memory_context_injection_enabled = True
            mock_config.memory_context_max_memories = 10

            context = await memory_service.get_context_for_prompt(
                user_id="user_123",
                current_query="",
            )

            assert "## User Context" in context
            mock_mem0_client.get_all.assert_called_once()

    @pytest.mark.unit
    async def test_get_context_when_injection_disabled(self, memory_service):
        """Should return empty string when injection disabled."""
        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True
            mock_config.memory_context_injection_enabled = False

            context = await memory_service.get_context_for_prompt(
                user_id="user_123",
                current_query="test",
            )

            assert context == ""

    @pytest.mark.unit
    async def test_get_context_when_no_memories(self, memory_service, mock_mem0_client):
        """Should return empty string when no memories found."""
        mock_mem0_client.search.return_value = {"results": []}

        with patch("seer.services.memory.user_memory.config") as mock_config:
            mock_config.memory_enabled = True
            mock_config.memory_context_injection_enabled = True
            mock_config.memory_context_max_memories = 10

            context = await memory_service.get_context_for_prompt(
                user_id="user_123",
                current_query="test",
            )

            assert context == ""

    @pytest.mark.unit
    async def test_format_truncates_long_memories(self, memory_service):
        """Should truncate very long memory text."""
        long_text = "x" * 300  # 300 chars
        memories = [{"memory": long_text}]

        formatted = memory_service._format_memories_for_prompt(memories)

        # Should be truncated to ~200 chars + "..."
        assert "..." in formatted
        # The full 300-char string should not appear in output
        assert long_text not in formatted


class TestSingleton:
    """Test singleton behavior."""

    @pytest.mark.unit
    def test_get_user_memory_service_returns_same_instance(self):
        """Should return the same service instance."""
        # Reset singleton
        import seer.services.memory.user_memory as module
        module._service_instance = None

        service1 = get_user_memory_service()
        service2 = get_user_memory_service()

        assert service1 is service2
