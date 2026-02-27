"""Unit tests for Redis session middleware."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.core.middleware.redis_sessions import (
    RedisSessionBackend,
    RedisSessionMiddleware,
    SessionInterface,
    SESSION_COOKIE_NAME,
)


@pytest.mark.unit
class TestSessionInterface:
    """Tests for the SessionInterface dict-like wrapper."""

    def test_basic_operations(self):
        """Test basic dict operations."""
        session = SessionInterface({"key1": "value1"})

        # Get
        assert session["key1"] == "value1"
        assert session.get("key1") == "value1"
        assert session.get("missing", "default") == "default"

        # Contains
        assert "key1" in session
        assert "missing" not in session

        # Len
        assert len(session) == 1

    def test_modification_tracking(self):
        """Test that modifications are tracked."""
        session = SessionInterface({})

        assert not session.is_modified

        # Set triggers modified
        session["key"] = "value"
        assert session.is_modified

    def test_modification_tracking_delete(self):
        """Test that deletion triggers modified."""
        session = SessionInterface({"key": "value"})

        assert not session.is_modified

        del session["key"]
        assert session.is_modified

    def test_modification_tracking_pop(self):
        """Test that pop triggers modified."""
        session = SessionInterface({"key": "value"})

        assert not session.is_modified

        result = session.pop("key")
        assert result == "value"
        assert session.is_modified

    def test_modification_tracking_update(self):
        """Test that update triggers modified."""
        session = SessionInterface({})

        assert not session.is_modified

        session.update({"key": "value"})
        assert session.is_modified
        assert session["key"] == "value"

    def test_modification_tracking_clear(self):
        """Test that clear triggers modified."""
        session = SessionInterface({"key": "value"})

        assert not session.is_modified

        session.clear()
        assert session.is_modified
        assert len(session) == 0

    def test_to_dict(self):
        """Test to_dict returns a copy."""
        session = SessionInterface({"key": "value"})

        result = session.to_dict()
        assert result == {"key": "value"}

        # Modifying result should not affect session
        result["key"] = "modified"
        assert session["key"] == "value"

    def test_iteration(self):
        """Test iteration over session."""
        session = SessionInterface({"a": 1, "b": 2})

        keys = list(session)
        assert sorted(keys) == ["a", "b"]

        items = list(session.items())
        assert sorted(items) == [("a", 1), ("b", 2)]


@pytest.mark.unit
@pytest.mark.asyncio
class TestRedisSessionBackend:
    """Tests for the Redis session backend."""

    async def test_read_empty_session_id(self):
        """Test reading with empty session ID returns empty dict."""
        backend = RedisSessionBackend("redis://localhost:6379/0")

        result = await backend.read("")
        assert result == {}

    async def test_read_success(self):
        """Test successful session read from Redis."""
        backend = RedisSessionBackend("redis://localhost:6379/0")

        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps({"user_id": "123"})

        with patch.object(backend, "_get_redis", return_value=mock_redis):
            result = await backend.read("test_session_id")

        assert result == {"user_id": "123"}
        mock_redis.get.assert_called_once_with("session:test_session_id")

    async def test_read_not_found(self):
        """Test reading non-existent session returns empty dict."""
        backend = RedisSessionBackend("redis://localhost:6379/0")

        mock_redis = AsyncMock()
        mock_redis.get.return_value = None

        with patch.object(backend, "_get_redis", return_value=mock_redis):
            result = await backend.read("nonexistent")

        assert result == {}

    async def test_read_handles_error(self):
        """Test read gracefully handles Redis errors."""
        backend = RedisSessionBackend("redis://localhost:6379/0")

        mock_redis = AsyncMock()
        # Use ConnectionError - a specific exception type the code handles
        mock_redis.get.side_effect = ConnectionError("Redis connection error")

        with patch.object(backend, "_get_redis", return_value=mock_redis):
            result = await backend.read("test_session_id")

        assert result == {}

    async def test_write_new_session(self):
        """Test writing a new session generates ID."""
        backend = RedisSessionBackend("redis://localhost:6379/0", session_ttl=600)

        mock_redis = AsyncMock()

        with patch.object(backend, "_get_redis", return_value=mock_redis):
            result = await backend.write("", {"user_id": "123"})

        # Should return a generated session ID
        assert len(result) > 0
        assert result != ""

        # Should call setex with TTL
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 600  # TTL
        assert json.loads(call_args[0][2]) == {"user_id": "123"}

    async def test_write_existing_session(self):
        """Test writing to existing session preserves ID."""
        backend = RedisSessionBackend("redis://localhost:6379/0", session_ttl=600)

        mock_redis = AsyncMock()

        with patch.object(backend, "_get_redis", return_value=mock_redis):
            result = await backend.write("existing_id", {"user_id": "456"})

        assert result == "existing_id"
        mock_redis.setex.assert_called_once_with(
            "session:existing_id",
            600,
            json.dumps({"user_id": "456"}),
        )

    async def test_delete_success(self):
        """Test successful session deletion."""
        backend = RedisSessionBackend("redis://localhost:6379/0")

        mock_redis = AsyncMock()

        with patch.object(backend, "_get_redis", return_value=mock_redis):
            await backend.delete("test_session_id")

        mock_redis.delete.assert_called_once_with("session:test_session_id")

    async def test_delete_empty_id(self):
        """Test deleting empty session ID does nothing."""
        backend = RedisSessionBackend("redis://localhost:6379/0")

        mock_redis = AsyncMock()

        with patch.object(backend, "_get_redis", return_value=mock_redis):
            await backend.delete("")

        mock_redis.delete.assert_not_called()


@pytest.mark.unit
class TestRedisSessionMiddlewareInit:
    """Tests for Redis session middleware initialization."""

    def test_default_config(self):
        """Test middleware default configuration."""
        app = MagicMock()
        middleware = RedisSessionMiddleware(
            app,
            redis_url="redis://localhost:6379/0",
        )

        assert middleware.cookie_name == SESSION_COOKIE_NAME
        assert middleware.session_ttl == 600
        assert middleware.cookie_httponly is True
        assert middleware.cookie_secure is False
        assert middleware.cookie_samesite == "lax"

    def test_custom_config(self):
        """Test middleware custom configuration."""
        app = MagicMock()
        middleware = RedisSessionMiddleware(
            app,
            redis_url="redis://localhost:6379/0",
            session_ttl=300,
            cookie_secure=True,
            cookie_samesite="strict",
        )

        assert middleware.session_ttl == 300
        assert middleware.cookie_secure is True
        assert middleware.cookie_samesite == "strict"
