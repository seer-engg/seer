"""
E2E tests for Workflow Execution API endpoints.

Tests workflow execution, run status tracking, and run history.
"""
import pytest
from httpx import AsyncClient


# =============================================================================
# Run Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_workflow_success(
    db_engine, authenticated_e2e_client: AsyncClient, simple_workflow_create_payload
):
    """Test successful workflow execution (runs from draft version)."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=simple_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Run workflow (uses draft version)
    run_payload = {
        "inputs": {
            "param1": "value1",
            "param2": 42
        }
    }
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/runs",
        json=run_payload
    )

    assert response.status_code == 201
    data = response.json()
    assert "run_id" in data
    assert data["workflow_id"] == workflow_id
    assert "status" in data
    assert data["status"] in ["pending", "running", "completed", "failed", "queued"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_workflow_without_inputs(
    db_engine, authenticated_e2e_client: AsyncClient, simple_workflow_create_payload
):
    """Test running workflow without inputs (runs from draft, no publish needed)."""
    # Create workflow (creates draft version)
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=simple_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Run without inputs (uses draft version)
    run_payload = {"inputs": {}}
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/runs",
        json=run_payload
    )

    assert response.status_code == 201
    data = response.json()
    assert "run_id" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_workflow_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test running non-existent workflow returns 404 or 400 for invalid ID format."""
    run_payload = {"inputs": {}}
    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/nonexistent_id/runs",
        json=run_payload
    )

    # ID validation happens before lookup, so may return 400 for invalid format
    assert response.status_code in [400, 404]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_workflow_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test running workflow without authentication returns 401."""
    run_payload = {"inputs": {}}
    response = await e2e_client.post(
        "/api/v1/workflows/any_id/runs",
        json=run_payload
    )

    assert response.status_code == 401


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_workflow_with_config(
    db_engine, authenticated_e2e_client: AsyncClient, simple_workflow_create_payload
):
    """Test running workflow with custom config."""
    # Create workflow (creates draft version)
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=simple_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Run with config (uses draft version)
    run_payload = {
        "inputs": {},
        "config": {
            "timeout": 300,
            "retry_on_error": True
        }
    }
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/runs",
        json=run_payload
    )

    assert response.status_code == 201


# =============================================================================
# List Workflow Runs Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_runs_empty(
    db_engine, authenticated_e2e_client: AsyncClient, simple_workflow_create_payload
):
    """Test listing runs when workflow has none."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=simple_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # List runs
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/runs"
    )

    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert data["runs"] == []


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_runs_with_items(
    db_engine, authenticated_e2e_client: AsyncClient, simple_workflow_create_payload
):
    """Test listing workflow runs returns executed runs."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=simple_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Execute workflow twice (uses draft version)
    run_payload = {"inputs": {}}
    await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/runs",
        json=run_payload
    )
    await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/runs",
        json=run_payload
    )

    # List runs
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/runs"
    )

    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert len(data["runs"]) == 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_runs_with_limit(
    db_engine, authenticated_e2e_client: AsyncClient, simple_workflow_create_payload
):
    """Test listing workflow runs with limit parameter."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=simple_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Execute 3 times (uses draft version)
    run_payload = {"inputs": {}}
    for _ in range(3):
        await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json=run_payload
        )

    # List with limit
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/runs?limit=2"
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["runs"]) == 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_runs_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test listing runs without authentication returns 401."""
    response = await e2e_client.get("/api/v1/workflows/any_id/runs")
    assert response.status_code == 401


# =============================================================================
# Get Run Status Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_run_status_success(
    db_engine, authenticated_e2e_client: AsyncClient, simple_workflow_create_payload
):
    """Test retrieving run status."""
    # Create and run workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=simple_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    run_response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/runs",
        json={"inputs": {}}
    )
    run_id = run_response.json()["run_id"]

    # Get run status
    response = await authenticated_e2e_client.get(f"/api/v1/runs/{run_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == run_id
    assert "status" in data
    assert data["status"] in ["pending", "running", "completed", "failed", "cancelled", "queued"]
    assert "created_at" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_run_status_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving non-existent run returns 404 or 400 for invalid ID format."""
    response = await authenticated_e2e_client.get("/api/v1/runs/nonexistent_run_id")

    # ID validation may happen before lookup
    assert response.status_code in [400, 404]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_run_status_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test retrieving run status without authentication returns 401."""
    response = await e2e_client.get("/api/v1/runs/any_run_id")
    assert response.status_code == 401


# =============================================================================
# Get Run History Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_run_history_success(
    db_engine, authenticated_e2e_client: AsyncClient, simple_workflow_create_payload
):
    """Test retrieving run history/execution trace."""
    # Create and run workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=simple_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    run_response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/runs",
        json={"inputs": {}}
    )
    run_id = run_response.json()["run_id"]

    # Get run history
    response = await authenticated_e2e_client.get(f"/api/v1/runs/{run_id}/history")

    assert response.status_code == 200
    data = response.json()
    # History format depends on implementation
    assert data is not None
    # May contain execution steps, state changes, etc.
    assert isinstance(data, dict)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_run_history_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving history for non-existent run returns 404 or 400 for invalid ID format."""
    response = await authenticated_e2e_client.get(
        "/api/v1/runs/nonexistent_run_id/history"
    )

    # ID validation may happen before lookup
    assert response.status_code in [400, 404]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_run_history_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test retrieving run history without authentication returns 401."""
    response = await e2e_client.get("/api/v1/runs/any_run_id/history")
    assert response.status_code == 401
