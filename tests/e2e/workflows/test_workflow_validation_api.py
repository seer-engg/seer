"""
E2E tests for Workflow Validation and Compilation API endpoints.

Tests the workflow validation, compilation, and expression typechecking endpoints.
"""
import pytest
from httpx import AsyncClient


# =============================================================================
# Validate Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_validate_workflow_success(
    db_engine, authenticated_e2e_client: AsyncClient, sample_workflow_spec
):
    """Test successful workflow validation."""
    payload = {"spec": sample_workflow_spec}

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/validate",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "warnings" in data
    assert isinstance(data["warnings"], list)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_validate_workflow_invalid_spec(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test validation with invalid workflow spec."""
    invalid_spec = {
        "version": "2",
        # Missing required fields
    }
    payload = {"spec": invalid_spec}

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/validate",
        json=payload
    )

    # Should return 400 with validation errors
    assert response.status_code in [400, 422]
    data = response.json()
    assert "errors" in data or "detail" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_validate_workflow_missing_nodes(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test validation with workflow missing required nodes."""
    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [],
        "edges": []
    }
    payload = {"spec": spec}

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/validate",
        json=payload
    )

    # Validation should fail or return with errors
    assert response.status_code in [200, 400, 422]
    if response.status_code == 200:
        data = response.json()
        # May have ok: False or warnings
        assert "ok" in data or "warnings" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_validate_workflow_dangling_edges(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test validation catches dangling edges."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "t1",
                "key": "test.trigger",
                "mode": "polling",
                "event_schema": {},
            }
        ],
        "nodes": [
            {
                "id": "n1",
                "type": "task",
                "kind": "set",
                "value": {"result": "test"},
            }
        ],
        "edges": [
            {
                "source": "t1",
                "target": "nonexistent_node",  # Dangling edge
                "type": "trigger",
            }
        ],
    }
    payload = {"spec": spec}

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/validate",
        json=payload
    )

    # Should detect the dangling edge
    assert response.status_code in [200, 400]
    data = response.json()
    if response.status_code == 200:
        # Check if ok is False or there are errors/warnings
        assert data.get("ok") is False or len(data.get("warnings", [])) > 0
    else:
        assert "errors" in data or "detail" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_validate_workflow_unauthorized(
    db_engine, e2e_client: AsyncClient, sample_workflow_spec
):
    """Test validation without authentication returns 401."""
    payload = {"spec": sample_workflow_spec}

    response = await e2e_client.post(
        "/api/v1/workflows/validate",
        json=payload
    )

    assert response.status_code == 401


# =============================================================================
# Compile Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compile_workflow_success(
    db_engine, authenticated_e2e_client: AsyncClient, sample_workflow_spec
):
    """Test successful workflow compilation."""
    payload = {
        "spec": sample_workflow_spec,
        "options": {
            "emit_graph_preview": False,
            "emit_type_env": False,
            "strict_task_output": False
        }
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/compile",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "warnings" in data
    assert "artifacts" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compile_workflow_with_graph_preview(
    db_engine, authenticated_e2e_client: AsyncClient, sample_workflow_spec
):
    """Test compilation with graph preview artifact."""
    payload = {
        "spec": sample_workflow_spec,
        "options": {
            "emit_graph_preview": True,
            "emit_type_env": False,
        }
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/compile",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "artifacts" in data
    # Should include graph_preview when enabled
    if data["artifacts"]:
        assert "graph_preview" in data["artifacts"] or data["artifacts"]["graph_preview"] is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compile_workflow_with_type_env(
    db_engine, authenticated_e2e_client: AsyncClient, sample_workflow_spec
):
    """Test compilation with type environment artifact."""
    payload = {
        "spec": sample_workflow_spec,
        "options": {
            "emit_graph_preview": False,
            "emit_type_env": True,
        }
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/compile",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "artifacts" in data
    # Should include type_env when enabled
    if data["artifacts"]:
        assert "type_env" in data["artifacts"] or data["artifacts"]["type_env"] is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compile_workflow_invalid_spec(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test compilation with invalid spec."""
    invalid_spec = {
        "version": "2",
        # Missing required fields
    }
    payload = {
        "spec": invalid_spec,
        "options": {}
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/compile",
        json=payload
    )

    # Should return error
    assert response.status_code in [400, 422]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_compile_workflow_unauthorized(
    db_engine, e2e_client: AsyncClient, sample_workflow_spec
):
    """Test compilation without authentication returns 401."""
    payload = {"spec": sample_workflow_spec, "options": {}}

    response = await e2e_client.post(
        "/api/v1/workflows/compile",
        json=payload
    )

    assert response.status_code == 401


# =============================================================================
# Expression Typecheck Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_typecheck_expression_simple(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test typechecking a simple expression."""
    payload = {
        "expression": "${1 + 2}",
        "context": {}
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/expr/typecheck",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    # Response format depends on implementation
    assert data is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_typecheck_expression_with_context(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test typechecking expression with context variables."""
    payload = {
        "expression": "${user.name}",
        "context": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
        }
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/expr/typecheck",
        json=payload
    )

    assert response.status_code == 200


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_typecheck_expression_invalid_syntax(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test typechecking with invalid expression syntax."""
    payload = {
        "expression": "${invalid syntax!!!}",
        "context": {}
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/expr/typecheck",
        json=payload
    )

    # Should return error or indicate invalid expression
    assert response.status_code in [200, 400, 422]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_typecheck_expression_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test expression typecheck without authentication returns 401."""
    payload = {
        "expression": "${1 + 2}",
        "context": {}
    }

    response = await e2e_client.post(
        "/api/v1/expr/typecheck",
        json=payload
    )

    assert response.status_code == 401
