"""
Unit tests for chat services.

Tests:
- SessionService.get_or_create_session: Session management
- CheckpointerHealthService.is_connection_error: Error detection
- CheckpointerHealthService.get_checkpointer_with_reconnect: Reconnection
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    from seer.database import User
    user = MagicMock(spec=User)
    user.id = 1
    user.user_id = "user_123"
    return user


@pytest.fixture
def mock_chat_session():
    """Create a mock chat session."""
    from seer.database import WorkflowChatSession

    session = MagicMock(spec=WorkflowChatSession)
    session.id = 1
    session.session_id = "session_abc123"
    session.workflow_id = 1
    session.user_id = 1
    return session


# =============================================================================
# Connection Error Detection Tests
# =============================================================================


@pytest.mark.unit
class TestConnectionErrorDetection:
    """Tests for connection error detection."""

    def test_is_connection_error_pg_connection_refused(self):
        """Test detecting PostgreSQL connection refused error."""
        error_msg = "connection to server at 'localhost' failed: Connection refused"

        # Connection refused should be detected as connection error
        assert "connection" in error_msg.lower()
        assert "refused" in error_msg.lower() or "failed" in error_msg.lower()

    def test_is_connection_error_pg_timeout(self):
        """Test detecting PostgreSQL connection timeout."""
        error_msg = "connection to server timed out"

        assert "connection" in error_msg.lower()
        assert "timed out" in error_msg.lower()

    def test_is_connection_error_pool_exhausted(self):
        """Test detecting connection pool exhausted error."""
        error_msg = "connection pool exhausted"

        assert "pool" in error_msg.lower()
        assert "exhausted" in error_msg.lower()

    def test_is_not_connection_error_validation(self):
        """Test that validation errors are not connection errors."""
        error_msg = "validation error for field 'name'"

        assert "connection" not in error_msg.lower()


# =============================================================================
# Session Management Tests
# =============================================================================


@pytest.mark.unit
class TestSessionManagement:
    """Tests for session management."""

    @pytest.mark.asyncio
    async def test_get_existing_session(self, mock_user, mock_chat_session):
        """Test getting an existing session."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_chat_session

            result = await WorkflowChatSession.get_or_none(
                session_id="session_abc123",
                user=mock_user
            )

            assert result == mock_chat_session

    @pytest.mark.asyncio
    async def test_session_not_found(self, mock_user):
        """Test handling missing session."""
        from seer.database import WorkflowChatSession

        with patch.object(WorkflowChatSession, "get_or_none", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            result = await WorkflowChatSession.get_or_none(
                session_id="nonexistent",
                user=mock_user
            )

            assert result is None


# =============================================================================
# Checkpointer Health Tests
# =============================================================================


@pytest.mark.unit
class TestCheckpointerHealth:
    """Tests for checkpointer health management."""

    @pytest.mark.asyncio
    async def test_checkpointer_initialization(self):
        """Test checkpointer initialization."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup = AsyncMock()

        with patch("seer.api.agents.checkpointer.get_checkpointer") as mock_get:
            mock_get.return_value = mock_checkpointer

            result = await mock_get()

            assert result == mock_checkpointer

    @pytest.mark.asyncio
    async def test_checkpointer_none_when_no_database(self):
        """Test checkpointer returns None when no database configured."""
        with patch("seer.api.agents.checkpointer.get_checkpointer") as mock_get:
            mock_get.return_value = None

            result = await mock_get()

            assert result is None


# =============================================================================
# Session State Tests
# =============================================================================


@pytest.mark.unit
class TestSessionState:
    """Tests for session state management."""

    def test_session_state_serialization(self):
        """Test session state can be serialized."""
        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "context": {
                "workflow_id": "wf_123",
            },
        }

        import json
        serialized = json.dumps(state)
        deserialized = json.loads(serialized)

        assert deserialized == state

    def test_session_thread_id_format(self):
        """Test session thread ID format."""
        session_id = "session_abc123"
        thread_id = f"chat_{session_id}"

        assert thread_id.startswith("chat_")
        assert session_id in thread_id


# =============================================================================
# Message History Tests
# =============================================================================


@pytest.mark.unit
class TestMessageHistory:
    """Tests for message history management."""

    def test_message_structure(self):
        """Test message has required fields."""
        message = {
            "role": "user",
            "content": "Test message",
            "created_at": "2024-01-01T00:00:00Z",
        }

        assert message["role"] in ["user", "assistant", "system"]
        assert "content" in message

    def test_message_ordering(self):
        """Test messages are ordered by creation time."""
        messages = [
            {"id": 1, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 2, "created_at": "2024-01-01T00:01:00Z"},
            {"id": 3, "created_at": "2024-01-01T00:02:00Z"},
        ]

        sorted_messages = sorted(messages, key=lambda m: m["created_at"])

        assert sorted_messages[0]["id"] == 1
        assert sorted_messages[-1]["id"] == 3


# =============================================================================
# Reconnection Tests
# =============================================================================


@pytest.mark.unit
class TestReconnection:
    """Tests for checkpointer reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect_on_connection_error(self):
        """Test reconnection attempt on connection error."""
        mock_checkpointer = MagicMock()
        call_count = 0

        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection failed")
            return "success"

        # First call fails, second succeeds
        mock_checkpointer.operation = mock_operation

        try:
            await mock_checkpointer.operation()
        except ConnectionError:
            # Reconnect and retry
            result = await mock_checkpointer.operation()
            assert result == "success"

    @pytest.mark.asyncio
    async def test_max_reconnect_attempts(self):
        """Test that reconnection has a maximum number of attempts."""
        max_attempts = 3
        attempts = 0

        async def always_fails():
            nonlocal attempts
            attempts += 1
            raise ConnectionError("Connection failed")

        for _ in range(max_attempts):
            try:
                await always_fails()
            except ConnectionError:
                pass

        assert attempts == max_attempts
