"""
Unit tests for api.tools.router module.

Tests the tool API endpoints with proper mocking.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# =============================================================================
# List Tools Endpoint Tests
# =============================================================================


@pytest.mark.unit
class TestListToolsEndpoint:
    """Tests for GET /api/tools endpoint."""

    @pytest.mark.asyncio
    async def test_list_tools_endpoint_returns_tools(self):
        """Test list_tools_endpoint returns tools list."""
        from seer.api.tools.router import list_tools_endpoint

        expected = {
            "tools": [
                {"name": "tool1", "description": "Tool 1"},
                {"name": "tool2", "description": "Tool 2"},
            ]
        }

        with patch("seer.api.tools.router.list_tools", new_callable=AsyncMock, return_value=expected):
            result = await list_tools_endpoint(integration_type=None)

        assert result == expected
        assert len(result["tools"]) == 2

    @pytest.mark.asyncio
    async def test_list_tools_endpoint_with_integration_filter(self):
        """Test list_tools_endpoint passes integration_type filter."""
        from seer.api.tools.router import list_tools_endpoint

        expected = {"tools": [{"name": "gmail_send", "integration_type": "gmail"}]}

        with patch("seer.api.tools.router.list_tools", new_callable=AsyncMock, return_value=expected) as mock_list:
            result = await list_tools_endpoint(integration_type="gmail")

        mock_list.assert_called_once_with(integration_type="gmail")
        assert result["tools"][0]["integration_type"] == "gmail"

    @pytest.mark.asyncio
    async def test_list_tools_endpoint_empty_result(self):
        """Test list_tools_endpoint handles empty result."""
        from seer.api.tools.router import list_tools_endpoint

        with patch("seer.api.tools.router.list_tools", new_callable=AsyncMock, return_value={"tools": []}):
            result = await list_tools_endpoint(integration_type="nonexistent")

        assert result["tools"] == []


# =============================================================================
# Execute Tool Endpoint Tests
# =============================================================================


@pytest.mark.unit
class TestExecuteToolEndpoint:
    """Tests for POST /api/tools/{tool_name}/execute endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request with user state."""
        request = MagicMock()
        request.state.db_user = MagicMock()
        request.state.db_user.user_id = "test_user_123"
        return request

    @pytest.fixture
    def mock_payload(self):
        """Create a mock execution payload."""
        from seer.api.tools.router import ExecuteToolRequest
        return ExecuteToolRequest(
            connection_id="conn_123",
            arguments={"param1": "value1"}
        )

    @pytest.mark.asyncio
    async def test_execute_tool_endpoint_success(self, mock_request, mock_payload):
        """Test successful tool execution via endpoint."""
        from seer.api.tools.router import execute_tool_endpoint

        expected = {"data": {"result": "success"}, "success": True}

        with patch("seer.api.tools.router.execute_tool_service", new_callable=AsyncMock, return_value=expected):
            result = await execute_tool_endpoint(
                request=mock_request,
                tool_name="test_tool",
                payload=mock_payload
            )

        assert result["success"] is True
        assert result["data"] == {"result": "success"}

    @pytest.mark.asyncio
    async def test_execute_tool_endpoint_passes_correct_args(self, mock_request, mock_payload):
        """Test endpoint passes correct arguments to service."""
        from seer.api.tools.router import execute_tool_endpoint

        with patch("seer.api.tools.router.execute_tool_service", new_callable=AsyncMock, return_value={"data": {}, "success": True}) as mock_exec:
            await execute_tool_endpoint(
                request=mock_request,
                tool_name="gmail_send",
                payload=mock_payload
            )

        mock_exec.assert_called_once_with(
            tool_name="gmail_send",
            user=mock_request.state.db_user,
            connection_id="conn_123",
            arguments={"param1": "value1"}
        )

    @pytest.mark.asyncio
    async def test_execute_tool_endpoint_handles_http_exception(self, mock_request, mock_payload):
        """Test endpoint returns error response on HTTPException."""
        from seer.api.tools.router import execute_tool_endpoint

        http_exc = HTTPException(status_code=403, detail="Insufficient permissions")

        with patch("seer.api.tools.router.execute_tool_service", new_callable=AsyncMock, side_effect=http_exc):
            result = await execute_tool_endpoint(
                request=mock_request,
                tool_name="test_tool",
                payload=mock_payload
            )

        assert result["success"] is False
        assert result["data"] is None
        assert result["error"] == "Insufficient permissions"

    @pytest.mark.asyncio
    async def test_execute_tool_endpoint_with_no_connection_id(self, mock_request):
        """Test endpoint handles payload without connection_id."""
        from seer.api.tools.router import execute_tool_endpoint, ExecuteToolRequest

        payload = ExecuteToolRequest(arguments={"key": "value"})
        expected = {"data": {"status": "ok"}, "success": True}

        with patch("seer.api.tools.router.execute_tool_service", new_callable=AsyncMock, return_value=expected) as mock_exec:
            result = await execute_tool_endpoint(
                request=mock_request,
                tool_name="no_auth_tool",
                payload=payload
            )

        mock_exec.assert_called_once_with(
            tool_name="no_auth_tool",
            user=mock_request.state.db_user,
            connection_id=None,
            arguments={"key": "value"}
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_endpoint_with_no_arguments(self, mock_request):
        """Test endpoint handles payload without arguments."""
        from seer.api.tools.router import execute_tool_endpoint, ExecuteToolRequest

        payload = ExecuteToolRequest(connection_id="conn_456")
        expected = {"data": {"items": []}, "success": True}

        with patch("seer.api.tools.router.execute_tool_service", new_callable=AsyncMock, return_value=expected) as mock_exec:
            result = await execute_tool_endpoint(
                request=mock_request,
                tool_name="list_items",
                payload=payload
            )

        mock_exec.assert_called_once_with(
            tool_name="list_items",
            user=mock_request.state.db_user,
            connection_id="conn_456",
            arguments=None
        )
        assert result["success"] is True


# =============================================================================
# Request/Response Model Tests
# =============================================================================


@pytest.mark.unit
class TestApiModels:
    """Tests for API request/response models."""

    def test_execute_tool_request_with_all_fields(self):
        """Test ExecuteToolRequest with all fields populated."""
        from seer.api.tools.router import ExecuteToolRequest

        request = ExecuteToolRequest(
            connection_id="conn_123",
            arguments={"param1": "value1", "param2": 42}
        )

        assert request.connection_id == "conn_123"
        assert request.arguments == {"param1": "value1", "param2": 42}

    def test_execute_tool_request_with_defaults(self):
        """Test ExecuteToolRequest with default values."""
        from seer.api.tools.router import ExecuteToolRequest

        request = ExecuteToolRequest()

        assert request.connection_id is None
        assert request.arguments is None

    def test_execute_tool_response_success(self):
        """Test ExecuteToolResponse for success case."""
        from seer.api.tools.router import ExecuteToolResponse

        response = ExecuteToolResponse(
            data={"result": "done"},
            success=True
        )

        assert response.data == {"result": "done"}
        assert response.success is True
        assert response.error is None

    def test_execute_tool_response_error(self):
        """Test ExecuteToolResponse for error case."""
        from seer.api.tools.router import ExecuteToolResponse

        response = ExecuteToolResponse(
            data=None,
            success=False,
            error="Tool execution failed"
        )

        assert response.data is None
        assert response.success is False
        assert response.error == "Tool execution failed"
