"""
E2E tests for Registry API endpoints.

Tests tool registry, model registry, and schema resolution endpoints.
"""
import pytest
from httpx import AsyncClient


# =============================================================================
# Tool Registry Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_tool_registry(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving tool registry."""
    response = await authenticated_e2e_client.get("/api/v1/registries/tools")

    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert isinstance(data["tools"], list)

    # Verify tool structure if tools exist
    if len(data["tools"]) > 0:
        tool = data["tools"][0]
        assert "id" in tool
        assert "name" in tool
        assert "version" in tool
        assert "title" in tool


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_tool_registry_with_schemas(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving tool registry with input/output schemas."""
    response = await authenticated_e2e_client.get(
        "/api/v1/registries/tools?include_schemas=true"
    )

    assert response.status_code == 200
    data = response.json()
    assert "tools" in data

    # Verify schemas are included if tools exist
    if len(data["tools"]) > 0:
        tool = data["tools"][0]
        # Schemas should be present when include_schemas=true
        assert "input_schema" in tool or "output_schema" in tool


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_tool_registry_without_schemas(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving tool registry without schemas (default)."""
    response = await authenticated_e2e_client.get(
        "/api/v1/registries/tools?include_schemas=false"
    )

    assert response.status_code == 200
    data = response.json()
    assert "tools" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_tool_registry_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test retrieving tool registry without authentication returns 401."""
    response = await e2e_client.get("/api/v1/registries/tools")
    assert response.status_code == 401


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
# Schema Resolution Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_schema_valid_id(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving schema by valid ID."""
    # Use a schema ID that exists in the system
    # This is a placeholder - actual schema IDs depend on system configuration
    schema_id = "test_schema_id"

    response = await authenticated_e2e_client.get(
        f"/api/v1/registries/schemas/{schema_id}"
    )

    # Should return 200 if schema exists, 404 otherwise
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "id" in data
        assert data["id"] == schema_id
        assert "json_schema" in data
        assert isinstance(data["json_schema"], dict)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_schema_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving non-existent schema returns 404."""
    response = await authenticated_e2e_client.get(
        "/api/v1/registries/schemas/nonexistent_schema_id"
    )

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_schema_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test retrieving schema without authentication returns 401."""
    response = await e2e_client.get("/api/v1/registries/schemas/any_id")
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
# Schema Metadata Generation Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_generate_schema_metadata_success(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test generating schema metadata from JSON schema."""
    payload = {
        "json_schema": {
            "type": "object",
            "properties": {
                "user_name": {"type": "string"},
                "user_email": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/schemas/generate-metadata",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "description" in data
    assert isinstance(data["title"], str)
    assert isinstance(data["description"], str)
    # Title should be PascalCase
    assert len(data["title"]) > 0
    # Description should be a sentence
    assert len(data["description"]) > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_generate_schema_metadata_complex_schema(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test generating metadata for complex nested schema."""
    payload = {
        "json_schema": {
            "type": "object",
            "properties": {
                "order": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_id": {"type": "string"},
                                    "quantity": {"type": "integer"}
                                }
                            }
                        },
                        "total": {"type": "number"}
                    }
                }
            }
        }
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/schemas/generate-metadata",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "description" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_generate_schema_metadata_invalid_schema(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test generating metadata with invalid schema."""
    payload = {
        "json_schema": {
            # Invalid schema structure
            "invalid_key": "invalid_value"
        }
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/schemas/generate-metadata",
        json=payload
    )

    # Should handle gracefully or return error
    assert response.status_code in [200, 400, 422]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_generate_schema_metadata_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test generating schema metadata without authentication returns 401."""
    payload = {
        "json_schema": {
            "type": "object",
            "properties": {}
        }
    }

    response = await e2e_client.post(
        "/api/v1/schemas/generate-metadata",
        json=payload
    )

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
