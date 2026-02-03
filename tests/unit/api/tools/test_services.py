"""
Unit tests for api.tools.services module.

Tests the tool listing and execution service functions with proper mocking.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# =============================================================================
# List Tools Tests
# =============================================================================


@pytest.mark.unit
class TestListTools:
    """Tests for list_tools function."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools_dict(self):
        """Test list_tools returns dictionary with tools list."""
        from seer.api.tools.services import list_tools

        mock_tool = MagicMock()
        mock_tool.get_metadata.return_value = {
            "name": "test_tool",
            "description": "A test tool",
            "required_scopes": [],
            "integration_type": "test",
        }

        with patch("seer.api.tools.services.get_tools_by_integration", return_value=[mock_tool.get_metadata()]):
            result = await list_tools()

        assert isinstance(result, dict)
        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_list_tools_filters_by_integration_type(self):
        """Test list_tools passes integration_type filter."""
        from seer.api.tools.services import list_tools

        mock_tool = MagicMock()
        mock_tool.get_metadata.return_value = {
            "name": "gmail_send",
            "description": "Send email via Gmail",
            "required_scopes": ["gmail.send"],
            "integration_type": "gmail",
        }

        with patch("seer.api.tools.services.get_tools_by_integration", return_value=[mock_tool.get_metadata()]) as mock_get:
            result = await list_tools(integration_type="gmail")

        mock_get.assert_called_once_with("gmail")
        assert len(result["tools"]) == 1
        assert result["tools"][0]["integration_type"] == "gmail"

    @pytest.mark.asyncio
    async def test_list_tools_returns_empty_list_when_no_tools(self):
        """Test list_tools returns empty list when no tools match."""
        from seer.api.tools.services import list_tools

        with patch("seer.api.tools.services.get_tools_by_integration", return_value=[]):
            result = await list_tools(integration_type="nonexistent")

        assert result["tools"] == []

    @pytest.mark.asyncio
    async def test_list_tools_raises_http_exception_on_error(self):
        """Test list_tools raises HTTPException on error."""
        from seer.api.tools.services import list_tools

        with patch("seer.api.tools.services.get_tools_by_integration", side_effect=ValueError("Registry error")):
            with pytest.raises(HTTPException) as exc_info:
                await list_tools()

        assert exc_info.value.status_code == 500
        assert "Error listing tools" in exc_info.value.detail


# =============================================================================
# Execute Tool Service Tests
# =============================================================================


@pytest.mark.unit
class TestExecuteToolService:
    """Tests for execute_tool_service function."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        user = MagicMock()
        user.user_id = "test_user_123"
        return user

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_user):
        """Test successful tool execution."""
        from seer.api.tools.services import execute_tool_service

        expected_result = {"message": "Email sent", "id": "msg_123"}

        with patch("seer.api.tools.services._execute_tool", new_callable=AsyncMock, return_value=expected_result):
            result = await execute_tool_service(
                tool_name="gmail_send",
                user=mock_user,
                arguments={"to": "test@example.com", "body": "Hello"}
            )

        assert result["success"] is True
        assert result["data"] == expected_result

    @pytest.mark.asyncio
    async def test_execute_tool_with_connection_id(self, mock_user):
        """Test tool execution with connection_id."""
        from seer.api.tools.services import execute_tool_service

        expected_result = {"status": "ok"}

        with patch("seer.api.tools.services._execute_tool", new_callable=AsyncMock, return_value=expected_result) as mock_exec:
            result = await execute_tool_service(
                tool_name="github_list_repos",
                user=mock_user,
                connection_id="conn_123",
                arguments={"per_page": 10}
            )

        mock_exec.assert_called_once_with(
            tool_name="github_list_repos",
            user=mock_user,
            connection_id="conn_123",
            arguments={"per_page": 10}
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_with_none_arguments(self, mock_user):
        """Test tool execution passes empty dict when arguments is None."""
        from seer.api.tools.services import execute_tool_service

        with patch("seer.api.tools.services._execute_tool", new_callable=AsyncMock, return_value={}) as mock_exec:
            await execute_tool_service(
                tool_name="test_tool",
                user=mock_user,
                arguments=None
            )

        mock_exec.assert_called_once_with(
            tool_name="test_tool",
            user=mock_user,
            connection_id=None,
            arguments={}
        )

    @pytest.mark.asyncio
    async def test_execute_tool_reraises_http_exception(self, mock_user):
        """Test execute_tool_service re-raises HTTPException."""
        from seer.api.tools.services import execute_tool_service

        http_exc = HTTPException(status_code=403, detail="Insufficient scopes")

        with patch("seer.api.tools.services._execute_tool", new_callable=AsyncMock, side_effect=http_exc):
            with pytest.raises(HTTPException) as exc_info:
                await execute_tool_service(
                    tool_name="gmail_send",
                    user=mock_user,
                    arguments={}
                )

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Insufficient scopes"

    @pytest.mark.asyncio
    async def test_execute_tool_wraps_general_exception(self, mock_user):
        """Test execute_tool_service wraps general exceptions in HTTPException."""
        from seer.api.tools.services import execute_tool_service

        with patch("seer.api.tools.services._execute_tool", new_callable=AsyncMock, side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(HTTPException) as exc_info:
                await execute_tool_service(
                    tool_name="test_tool",
                    user=mock_user,
                    arguments={}
                )

        assert exc_info.value.status_code == 500
        assert "Error executing tool" in exc_info.value.detail
