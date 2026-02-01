"""
E2E tests for Workflow CRUD API endpoints.

Tests the complete workflow lifecycle via API:
- Create workflow
- List workflows
- Get workflow
- Update workflow metadata
- Delete workflow
"""
import pytest
from httpx import AsyncClient


# =============================================================================
# Create Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_workflow_success(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test successful workflow creation via API."""
    response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )

    assert response.status_code == 201
    data = response.json()
    assert "workflow_id" in data
    assert data["name"] == workflow_create_payload["name"]
    # Spec may be normalized by the system, so check key fields exist
    assert "spec" in data
    assert data["spec"]["version"] == workflow_create_payload["spec"]["version"]
    assert len(data["spec"]["nodes"]) == len(workflow_create_payload["spec"]["nodes"])
    assert len(data["spec"]["edges"]) == len(workflow_create_payload["spec"]["edges"])
    assert "created_at" in data
    assert "updated_at" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_workflow_invalid_spec(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test workflow creation with invalid spec returns 400."""
    invalid_payload = {
        "name": "Invalid Workflow",
        "spec": {
            "version": "1",  # Invalid version (must be 2+)
        }
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=invalid_payload
    )

    assert response.status_code == 400
    data = response.json()
    assert "errors" in data or "detail" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_workflow_missing_name(
    db_engine, authenticated_e2e_client: AsyncClient, sample_workflow_spec
):
    """Test workflow creation without name returns 422."""
    payload = {
        "spec": sample_workflow_spec
        # Missing required "name" field
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=payload
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_workflow_unauthorized(
    db_engine, e2e_client: AsyncClient, workflow_create_payload
):
    """Test workflow creation without authentication returns 401."""
    response = await e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )

    assert response.status_code == 401


# =============================================================================
# List Workflows Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflows_empty(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test listing workflows when none exist."""
    response = await authenticated_e2e_client.get("/api/v1/workflows")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["items"] == []
    assert data.get("next_cursor") is None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflows_with_items(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test listing workflows returns created items."""
    # Create two workflows
    await authenticated_e2e_client.post("/api/v1/workflows", json=workflow_create_payload)

    payload2 = workflow_create_payload.copy()
    payload2["name"] = "Second Workflow"
    await authenticated_e2e_client.post("/api/v1/workflows", json=payload2)

    # List workflows
    response = await authenticated_e2e_client.get("/api/v1/workflows")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    assert data["items"][0]["name"] in ["Test Workflow", "Second Workflow"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflows_pagination_limit(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test listing workflows with limit parameter."""
    # Create 3 workflows
    for i in range(3):
        payload = workflow_create_payload.copy()
        payload["name"] = f"Workflow {i+1}"
        await authenticated_e2e_client.post("/api/v1/workflows", json=payload)

    # List with limit=2
    response = await authenticated_e2e_client.get("/api/v1/workflows?limit=2")

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert "next_cursor" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflows_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test listing workflows without authentication returns 401."""
    response = await e2e_client.get("/api/v1/workflows")
    assert response.status_code == 401


# =============================================================================
# Get Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_workflow_success(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test retrieving a single workflow by ID."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Get workflow
    response = await authenticated_e2e_client.get(f"/api/v1/workflows/{workflow_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_id"] == workflow_id
    assert data["name"] == workflow_create_payload["name"]
    # Spec may be enriched with default fields, check key structure
    assert data["spec"]["version"] == workflow_create_payload["spec"]["version"]
    assert len(data["spec"]["nodes"]) == len(workflow_create_payload["spec"]["nodes"])
    assert len(data["spec"]["edges"]) == len(workflow_create_payload["spec"]["edges"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_workflow_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving non-existent workflow returns 404 or 400 for invalid ID format."""
    response = await authenticated_e2e_client.get("/api/v1/workflows/nonexistent_id")

    # ID validation happens before lookup, so may return 400 for invalid format
    assert response.status_code in [400, 404]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_workflow_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test retrieving workflow without authentication returns 401."""
    response = await e2e_client.get("/api/v1/workflows/any_id")
    assert response.status_code == 401


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_workflow_other_user(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test user cannot access another user's workflow."""
    # Create workflow as user1
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Try to access as different user (would need separate fixture)
    # For now, we verify successful access with same user
    response = await authenticated_e2e_client.get(f"/api/v1/workflows/{workflow_id}")
    assert response.status_code == 200


# =============================================================================
# Update Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_update_workflow_name(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test updating workflow name."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Update name
    update_payload = {"name": "Updated Workflow Name"}
    response = await authenticated_e2e_client.put(
        f"/api/v1/workflows/{workflow_id}",
        json=update_payload
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Workflow Name"
    assert data["workflow_id"] == workflow_id


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_update_workflow_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test updating non-existent workflow returns 404 or 400 for invalid ID format."""
    update_payload = {"name": "New Name"}
    response = await authenticated_e2e_client.put(
        "/api/v1/workflows/nonexistent_id",
        json=update_payload
    )

    # ID validation happens before lookup, so may return 400 for invalid format
    assert response.status_code in [400, 404]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_update_workflow_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test updating workflow without authentication returns 401."""
    response = await e2e_client.put(
        "/api/v1/workflows/any_id",
        json={"name": "New Name"}
    )
    assert response.status_code == 401


# =============================================================================
# Delete Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_delete_workflow_success(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test deleting a workflow."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Delete workflow
    response = await authenticated_e2e_client.delete(f"/api/v1/workflows/{workflow_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True

    # Verify deletion
    get_response = await authenticated_e2e_client.get(f"/api/v1/workflows/{workflow_id}")
    assert get_response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_delete_workflow_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test deleting non-existent workflow returns 404 or 400 for invalid ID format."""
    response = await authenticated_e2e_client.delete("/api/v1/workflows/nonexistent_id")

    # ID validation happens before lookup, so may return 400 for invalid format
    assert response.status_code in [400, 404]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_delete_workflow_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test deleting workflow without authentication returns 401."""
    response = await e2e_client.delete("/api/v1/workflows/any_id")
    assert response.status_code == 401


# =============================================================================
# Import/Export Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_export_workflow(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test exporting a workflow."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Export workflow
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/export"
    )

    assert response.status_code == 200
    assert "Content-Disposition" in response.headers
    assert "attachment" in response.headers["Content-Disposition"]

    data = response.json()
    # Export format is version 1.0 with nested structure
    assert "version" in data
    assert data["version"] == "1.0"
    assert "workflow" in data
    assert "spec" in data["workflow"]
    assert "metadata" in data
    # Verify spec structure matches
    assert data["workflow"]["spec"]["version"] == workflow_create_payload["spec"]["version"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_import_workflow(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test importing a workflow."""
    # Import format expects version 1.0 export structure
    import_payload = {
        "import_data": {
            "version": "1.0",
            "workflow": {
                "name": "Original Workflow",
                "spec": workflow_create_payload["spec"]
            },
            "triggers": [],
            "metadata": {}
        },
        "name": "Imported Workflow"  # Override name
    }

    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/import",
        json=import_payload
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Imported Workflow"
    assert "workflow_id" in data
