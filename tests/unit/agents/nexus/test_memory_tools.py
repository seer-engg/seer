"""
Unit tests for Nexus memory tools.

Tests the agent tools that allow searching and retrieving user memories.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from seer.agents.nexus.tools.memory_tools import (
    recall_memories,
    search_past_sessions,
    get_user_profile,
    _format_memories_for_tool_response,
    _format_session_search_results,
)


@pytest.fixture
def mock_user():
    """Create a mock User object."""
    user = MagicMock()
    user.user_id = "clerk_user_123"
    return user


@pytest.fixture
def sample_memories():
    """Sample memory results."""
    return [
        {
            "id": "mem_1",
            "memory": "User prefers Slack notifications over email",
            "score": 0.95,
            "metadata": {"session_id": 1, "session_title": "Slack Integration"},
        },
        {
            "id": "mem_2",
            "memory": "User's company uses PostgreSQL",
            "score": 0.82,
            "metadata": {"workflow_id": 42},
        },
    ]


class TestFormatMemoriesForToolResponse:
    """Test memory formatting for tool responses."""

    @pytest.mark.unit
    def test_format_with_memories(self, sample_memories):
        """Should format memories with metadata."""
        result = _format_memories_for_tool_response(sample_memories)

        assert "Found 2 relevant memories" in result
        assert "User prefers Slack notifications" in result
        assert "Session: Slack Integration" in result
        assert "Workflow: wf_42" in result
        assert "Relevance: 0.95" in result

    @pytest.mark.unit
    def test_format_empty_list(self):
        """Should return message when no memories."""
        result = _format_memories_for_tool_response([])

        assert "No relevant memories found" in result


class TestFormatSessionSearchResults:
    """Test session search results formatting."""

    @pytest.mark.unit
    def test_format_with_session_memories(self):
        """Should group memories by session."""
        memories = [
            {"id": "1", "memory": "Created Slack workflow", "metadata": {"session_id": 1, "session_title": "Slack Setup"}},
            {"id": "2", "memory": "Added notification trigger", "metadata": {"session_id": 1, "session_title": "Slack Setup"}},
            {"id": "3", "memory": "Built Gmail integration", "metadata": {"session_id": 2, "session_title": "Gmail Project"}},
        ]

        result = _format_session_search_results(memories)

        assert "Found matches in 2 past session(s)" in result
        assert "Slack Setup" in result
        assert "Gmail Project" in result

    @pytest.mark.unit
    def test_format_empty_sessions(self):
        """Should handle empty results."""
        result = _format_session_search_results([])

        assert "No matching past sessions found" in result


class TestRecallMemories:
    """Test recall_memories tool."""

    @pytest.mark.unit
    async def test_recall_memories_success(self, mock_user, sample_memories):
        """Should search and format memories."""
        with patch("seer.agents.nexus.tools.memory_tools.config") as mock_config:
            mock_config.memory_enabled = True

            with patch("seer.agents.nexus.tools.memory_tools._current_thread_id") as mock_thread:
                mock_thread.get.return_value = "thread_123"

                with patch("seer.agents.nexus.tools.memory_tools.get_user_for_thread", new_callable=AsyncMock) as mock_get_user:
                    mock_get_user.return_value = mock_user

                    with patch("seer.agents.nexus.tools.memory_tools.UserMemoryService") as MockService:
                        mock_service = MockService.return_value
                        mock_service.search = AsyncMock(return_value=sample_memories)

                        # Call the tool's underlying function
                        result = await recall_memories.ainvoke({"query": "notifications", "limit": 5})

                        assert "Found 2 relevant memories" in result
                        mock_service.search.assert_called_once()

    @pytest.mark.unit
    async def test_recall_memories_disabled(self):
        """Should return message when memory disabled."""
        with patch("seer.agents.nexus.tools.memory_tools.config") as mock_config:
            mock_config.memory_enabled = False

            result = await recall_memories.ainvoke({"query": "test", "limit": 5})

            assert "not enabled" in result

    @pytest.mark.unit
    async def test_recall_memories_no_thread(self):
        """Should handle missing thread context."""
        with patch("seer.agents.nexus.tools.memory_tools.config") as mock_config:
            mock_config.memory_enabled = True

            with patch("seer.agents.nexus.tools.memory_tools._current_thread_id") as mock_thread:
                mock_thread.get.return_value = None

                result = await recall_memories.ainvoke({"query": "test", "limit": 5})

                assert "Unable to identify" in result


class TestSearchPastSessions:
    """Test search_past_sessions tool."""

    @pytest.mark.unit
    async def test_search_sessions_success(self, mock_user):
        """Should search sessions and format results."""
        memories = [
            {"id": "1", "memory": "Built Slack workflow", "metadata": {"session_id": 1, "session_title": "Slack Setup"}},
        ]

        with patch("seer.agents.nexus.tools.memory_tools.config") as mock_config:
            mock_config.memory_enabled = True

            with patch("seer.agents.nexus.tools.memory_tools._current_thread_id") as mock_thread:
                mock_thread.get.return_value = "thread_123"

                with patch("seer.agents.nexus.tools.memory_tools.get_user_for_thread", new_callable=AsyncMock) as mock_get_user:
                    mock_get_user.return_value = mock_user

                    with patch("seer.agents.nexus.tools.memory_tools.UserMemoryService") as MockService:
                        mock_service = MockService.return_value
                        mock_service.search = AsyncMock(return_value=memories)

                        result = await search_past_sessions.ainvoke({"query": "Slack", "limit": 3})

                        assert "Slack Setup" in result
                        # Verify filter was applied
                        call_kwargs = mock_service.search.call_args[1]
                        assert call_kwargs["filters"]["has_session_id"] is True


class TestGetUserProfile:
    """Test get_user_profile tool."""

    @pytest.mark.unit
    async def test_get_profile_success(self, mock_user):
        """Should get all memories and format as profile."""
        all_memories = [
            {"memory": "Prefers Slack"},
            {"memory": "Uses PostgreSQL"},
            {"memory": "Works at TechCorp"},
        ]

        with patch("seer.agents.nexus.tools.memory_tools.config") as mock_config:
            mock_config.memory_enabled = True

            with patch("seer.agents.nexus.tools.memory_tools._current_thread_id") as mock_thread:
                mock_thread.get.return_value = "thread_123"

                with patch("seer.agents.nexus.tools.memory_tools.get_user_for_thread", new_callable=AsyncMock) as mock_get_user:
                    mock_get_user.return_value = mock_user

                    with patch("seer.agents.nexus.tools.memory_tools.UserMemoryService") as MockService:
                        mock_service = MockService.return_value
                        mock_service.get_all = AsyncMock(return_value=all_memories)

                        result = await get_user_profile.ainvoke({})

                        assert "User Profile" in result
                        assert "3 stored memories" in result
                        assert "Prefers Slack" in result

    @pytest.mark.unit
    async def test_get_profile_new_user(self, mock_user):
        """Should indicate when no memories exist."""
        with patch("seer.agents.nexus.tools.memory_tools.config") as mock_config:
            mock_config.memory_enabled = True

            with patch("seer.agents.nexus.tools.memory_tools._current_thread_id") as mock_thread:
                mock_thread.get.return_value = "thread_123"

                with patch("seer.agents.nexus.tools.memory_tools.get_user_for_thread", new_callable=AsyncMock) as mock_get_user:
                    mock_get_user.return_value = mock_user

                    with patch("seer.agents.nexus.tools.memory_tools.UserMemoryService") as MockService:
                        mock_service = MockService.return_value
                        mock_service.get_all = AsyncMock(return_value=[])

                        result = await get_user_profile.ainvoke({})

                        assert "No stored memories" in result
                        assert "new user" in result
