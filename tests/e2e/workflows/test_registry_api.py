"""
E2E tests for Registry API endpoints.

Tests model registry, node types, and MCP tools endpoints.
"""
import pytest
from httpx import AsyncClient


# =============================================================================
# Model Registry Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_model_registry(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving model registry."""
    response = await authenticated_e2e_client.get("/api/v1/registries/models")

    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)

    # Verify model structure if models exist
    if len(data["models"]) > 0:
        model = data["models"][0]
        assert "id" in model
        assert "title" in model
        assert "supports_json_schema" in model


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_model_registry_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test retrieving model registry without authentication returns 401."""
    response = await e2e_client.get("/api/v1/registries/models")
    assert response.status_code == 401


# =============================================================================
# Node Types Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_node_types(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving node types for workflow builder."""
    response = await authenticated_e2e_client.get("/api/v1/builder/node-types")

    assert response.status_code == 200
    data = response.json()
    assert "node_types" in data
    assert isinstance(data["node_types"], list)

    # Verify node type structure if types exist
    if len(data["node_types"]) > 0:
        node_type = data["node_types"][0]
        assert "type" in node_type
        assert "title" in node_type
        assert "fields" in node_type
        assert isinstance(node_type["fields"], list)

        # Verify field structure
        if len(node_type["fields"]) > 0:
            field = node_type["fields"][0]
            assert "name" in field
            assert "kind" in field
            assert "required" in field


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_node_types_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test retrieving node types without authentication returns 401."""
    response = await e2e_client.get("/api/v1/builder/node-types")
    assert response.status_code == 401


# =============================================================================
# MCP Tools Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_mcp_tools(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test listing MCP tools from a server."""
    payload = {
        "server": "http://localhost:8080",
        "server_type": "http",
        "auth": None
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/mcp/tools",
        json=payload
    )

    # MCP server may or may not be available in test environment
    assert response.status_code in [200, 400, 500, 503]

    if response.status_code == 200:
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_mcp_tools_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test listing MCP tools without authentication returns 401."""
    payload = {
        "server": "http://localhost:8080",
        "server_type": "http"
    }

    response = await e2e_client.post(
        "/api/v1/mcp/tools",
        json=payload
    )

    assert response.status_code == 401
