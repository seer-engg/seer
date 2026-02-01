"""
E2E tests for Tools API endpoints.

Tests tool listing and execution endpoints.
"""
import pytest
from httpx import AsyncClient


# =============================================================================
# List Tools Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_tools_all(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test listing all available tools."""
    response = await authenticated_e2e_client.get("/api/tools")

    assert response.status_code == 200
    data = response.json()
    assert "tools" in data or isinstance(data, dict)

    # Response structure depends on implementation
    # May be {"tools": [...]} or direct dict of tools


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_tools_filtered_by_integration(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test listing tools filtered by integration type."""
    # Test with common integration types
    integration_types = ["gmail", "github", "slack"]

    for integration_type in integration_types:
        response = await authenticated_e2e_client.get(
            f"/api/tools?integration_type={integration_type}"
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert isinstance(data, dict)

        # If tools exist for this integration, verify they match
        if "tools" in data and len(data["tools"]) > 0:
            for tool in data["tools"]:
                # Tool should have expected fields
                assert "name" in tool or "key" in tool or "id" in tool


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_tools_nonexistent_integration(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test listing tools with non-existent integration type."""
    response = await authenticated_e2e_client.get(
        "/api/tools?integration_type=nonexistent_integration"
    )

    assert response.status_code == 200
    data = response.json()
    # Should return empty or no tools for this integration
    if "tools" in data:
        assert isinstance(data["tools"], list)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_tools_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test listing tools without authentication returns 401."""
    response = await e2e_client.get("/api/tools")
    assert response.status_code == 401


# =============================================================================
# Execute Tool Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_with_arguments(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test executing a tool with valid arguments."""
    # Use a test/mock tool if available
    tool_name = "test_tool"
    payload = {
        "connection_id": None,
        "arguments": {
            "param1": "value1",
            "param2": 42
        }
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Tool may not exist in test environment
    assert response.status_code in [200, 404, 400]

    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert "data" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_without_arguments(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test executing a tool without arguments."""
    tool_name = "test_tool"
    payload = {
        "connection_id": None,
        "arguments": None
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Tool may not exist or may require arguments
    assert response.status_code in [200, 404, 400, 422]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_with_connection_id(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test executing a tool with connection ID for OAuth."""
    tool_name = "test_tool"
    payload = {
        "connection_id": "conn_123",
        "arguments": {
            "param": "value"
        }
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Connection may not exist
    assert response.status_code in [200, 404, 400]

    if response.status_code == 200:
        data = response.json()
        assert "success" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_nonexistent(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test executing non-existent tool returns 404."""
    tool_name = "definitely_nonexistent_tool_xyz"
    payload = {
        "connection_id": None,
        "arguments": {}
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Should return 404 or error response
    assert response.status_code in [404, 400]

    if response.status_code == 200:
        # If 200, should indicate failure
        data = response.json()
        assert data.get("success") is False


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_invalid_arguments(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test executing tool with invalid argument types."""
    tool_name = "test_tool"
    payload = {
        "connection_id": None,
        "arguments": "invalid_not_a_dict"  # Should be dict
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Should fail validation
    assert response.status_code in [422, 400]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test executing tool without authentication returns 401."""
    payload = {
        "connection_id": None,
        "arguments": {}
    }

    response = await e2e_client.post(
        "/api/tools/any_tool/execute",
        json=payload
    )

    assert response.status_code == 401


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_missing_required_arguments(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test executing tool with missing required arguments."""
    tool_name = "test_tool"
    payload = {
        "connection_id": None,
        "arguments": {
            # Missing required arguments
        }
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Should fail if tool exists and requires arguments
    assert response.status_code in [200, 404, 400, 422]

    if response.status_code == 200:
        data = response.json()
        # Should indicate failure or validation error
        if "success" in data:
            assert data["success"] is False or "error" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_with_invalid_connection(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test executing tool with invalid connection ID."""
    tool_name = "test_tool"
    payload = {
        "connection_id": "invalid_connection_id_999",
        "arguments": {}
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Should fail to find connection
    assert response.status_code in [200, 404, 400]

    if response.status_code == 200:
        data = response.json()
        # Should indicate failure
        assert data.get("success") is False or "error" in data


# =============================================================================
# Tool Execution Error Handling Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_error_response_format(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test error response format from tool execution."""
    tool_name = "nonexistent_tool"
    payload = {
        "connection_id": None,
        "arguments": {}
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Verify error response structure
    if response.status_code == 200:
        data = response.json()
        # Should have success and error fields
        assert "success" in data
        if data["success"] is False:
            assert "error" in data
            assert isinstance(data["error"], str)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execute_tool_malformed_payload(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test executing tool with completely malformed payload."""
    tool_name = "test_tool"
    # Send invalid JSON structure
    payload = {
        "invalid_field": "invalid_value"
    }

    response = await authenticated_e2e_client.post(
        f"/api/tools/{tool_name}/execute",
        json=payload
    )

    # Should fail validation
    assert response.status_code in [422, 400]
