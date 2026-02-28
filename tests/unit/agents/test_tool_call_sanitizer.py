"""Tests for ToolCallSanitizationMiddleware."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from seer.agents.nexus.tool_call_sanitizer import ToolCallSanitizationMiddleware


class TestToolCallSanitizationMiddleware:
    """Tests for the ToolCallSanitizationMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a middleware instance for testing."""
        return ToolCallSanitizationMiddleware()

    @pytest.mark.asyncio
    async def test_strips_leading_whitespace_from_tool_call_ids(self, middleware):
        """Test that leading whitespace is stripped from IDs."""
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"id": " list_tools:8", "name": "list_tools", "args": {}},
            {"id": "  functions.search:9", "name": "search", "args": {}},
            {"id": " functions.list_available_triggers:13", "name": "list_triggers", "args": {}},
        ]

        handler = AsyncMock(return_value=mock_response)

        result = await middleware.awrap_model_call({}, handler)

        # Whitespace stripped, colons and periods preserved
        assert result.tool_calls[0]["id"] == "list_tools:8"
        assert result.tool_calls[1]["id"] == "functions.search:9"
        assert result.tool_calls[2]["id"] == "functions.list_available_triggers:13"

    @pytest.mark.asyncio
    async def test_preserves_ids_without_whitespace(self, middleware):
        """Test that IDs without whitespace are not modified."""
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"id": "call_abc123", "name": "tool", "args": {}},
            {"id": "functions.tool:456", "name": "another_tool", "args": {}},
        ]

        handler = AsyncMock(return_value=mock_response)

        result = await middleware.awrap_model_call({}, handler)

        assert result.tool_calls[0]["id"] == "call_abc123"
        assert result.tool_calls[1]["id"] == "functions.tool:456"

    @pytest.mark.asyncio
    async def test_handles_response_without_tool_calls_attribute(self, middleware):
        """Test handling of responses without tool_calls attribute."""
        mock_response = MagicMock(spec=[])  # No attributes
        del mock_response.tool_calls  # Ensure no tool_calls

        handler = AsyncMock(return_value=mock_response)

        result = await middleware.awrap_model_call({}, handler)

        assert result == mock_response

    @pytest.mark.asyncio
    async def test_handles_response_with_none_tool_calls(self, middleware):
        """Test handling of responses where tool_calls is None."""
        mock_response = MagicMock()
        mock_response.tool_calls = None

        handler = AsyncMock(return_value=mock_response)

        result = await middleware.awrap_model_call({}, handler)

        assert result == mock_response
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_handles_response_with_empty_tool_calls(self, middleware):
        """Test handling of responses with empty tool_calls list."""
        mock_response = MagicMock()
        mock_response.tool_calls = []

        handler = AsyncMock(return_value=mock_response)

        result = await middleware.awrap_model_call({}, handler)

        assert result == mock_response
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_calls_handler_with_request(self, middleware):
        """Test that the handler is called with the original request."""
        mock_request = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_response = MagicMock()
        mock_response.tool_calls = None

        handler = AsyncMock(return_value=mock_response)

        await middleware.awrap_model_call(mock_request, handler)

        handler.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_mixed_ids_with_and_without_whitespace(self, middleware):
        """Test handling of mix of IDs with and without whitespace."""
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"id": "valid_id_123", "name": "tool1", "args": {}},  # No whitespace
            {"id": " has_leading_space:8", "name": "tool2", "args": {}},  # Has whitespace
            {"id": "another.valid:id", "name": "tool3", "args": {}},  # No whitespace
        ]

        handler = AsyncMock(return_value=mock_response)

        result = await middleware.awrap_model_call({}, handler)

        assert result.tool_calls[0]["id"] == "valid_id_123"  # Unchanged
        assert result.tool_calls[1]["id"] == "has_leading_space:8"  # Whitespace stripped
        assert result.tool_calls[2]["id"] == "another.valid:id"  # Unchanged

    @pytest.mark.asyncio
    async def test_preserves_other_tool_call_fields(self, middleware):
        """Test that sanitization doesn't affect other tool_call fields."""
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {
                "id": " bad:id:1",
                "name": "my_tool",
                "args": {"param1": "value1", "param2": 42},
            },
        ]

        handler = AsyncMock(return_value=mock_response)

        result = await middleware.awrap_model_call({}, handler)

        tc = result.tool_calls[0]
        assert tc["id"] == "bad:id:1"  # Whitespace stripped, colons preserved
        assert tc["name"] == "my_tool"  # Preserved
        assert tc["args"] == {"param1": "value1", "param2": 42}  # Preserved
