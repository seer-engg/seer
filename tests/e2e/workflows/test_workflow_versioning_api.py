"""
E2E tests for Workflow Versioning and Publishing API endpoints.

Tests draft management, publishing, and version restore functionality.
"""
import pytest
from httpx import AsyncClient


# =============================================================================
# Draft Management Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_patch_workflow_draft(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload, sample_workflow_spec
):
    """Test patching workflow draft spec."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Modify spec
    modified_spec = sample_workflow_spec.copy()
    modified_spec["nodes"][0]["value"] = {"result": "modified"}

    # Patch draft
    patch_payload = {"spec": modified_spec}
    response = await authenticated_e2e_client.patch(
        f"/api/v1/workflows/{workflow_id}/draft",
        json=patch_payload
    )

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_id"] == workflow_id
    assert data["spec"]["nodes"][0]["value"] == {"result": "modified"}


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_patch_workflow_draft_invalid_spec(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test patching draft with invalid spec returns error."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Invalid spec
    invalid_spec = {
        "version": "2",
        # Missing required fields
    }
    patch_payload = {"spec": invalid_spec}

    response = await authenticated_e2e_client.patch(
        f"/api/v1/workflows/{workflow_id}/draft",
        json=patch_payload
    )

    assert response.status_code in [400, 422]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_patch_workflow_draft_not_found(
    db_engine, authenticated_e2e_client: AsyncClient, sample_workflow_spec
):
    """Test patching draft for non-existent workflow returns 404."""
    patch_payload = {"spec": sample_workflow_spec}

    response = await authenticated_e2e_client.patch(
        "/api/v1/workflows/nonexistent_id/draft",
        json=patch_payload
    )

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_patch_workflow_draft_unauthorized(
    db_engine, e2e_client: AsyncClient, sample_workflow_spec
):
    """Test patching draft without authentication returns 401."""
    patch_payload = {"spec": sample_workflow_spec}

    response = await e2e_client.patch(
        "/api/v1/workflows/any_id/draft",
        json=patch_payload
    )

    assert response.status_code == 401


# =============================================================================
# Publish Workflow Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_publish_workflow_success(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test successful workflow publishing."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Publish workflow
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/publish",
        json={}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_id"] == workflow_id
    # May include version info or published status


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_publish_workflow_multiple_times(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload, sample_workflow_spec
):
    """Test publishing workflow multiple times creates versions."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Publish first version
    await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/publish",
        json={}
    )

    # Modify draft
    modified_spec = sample_workflow_spec.copy()
    modified_spec["nodes"][0]["value"] = {"result": "v2"}
    await authenticated_e2e_client.patch(
        f"/api/v1/workflows/{workflow_id}/draft",
        json={"spec": modified_spec}
    )

    # Publish second version
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/publish",
        json={}
    )

    assert response.status_code == 200

    # Check versions list
    versions_response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/versions"
    )
    assert versions_response.status_code == 200
    versions_data = versions_response.json()
    assert len(versions_data["versions"]) >= 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_publish_workflow_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test publishing non-existent workflow returns 404."""
    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/nonexistent_id/publish",
        json={}
    )

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_publish_workflow_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test publishing workflow without authentication returns 401."""
    response = await e2e_client.post(
        "/api/v1/workflows/any_id/publish",
        json={}
    )

    assert response.status_code == 401


# =============================================================================
# List Workflow Versions Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_versions_empty(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test listing versions for unpublished workflow."""
    # Create workflow (not published)
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # List versions
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/versions"
    )

    assert response.status_code == 200
    data = response.json()
    assert "versions" in data
    assert data["workflow_id"] == workflow_id
    # May have draft version or empty list
    assert isinstance(data["versions"], list)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_versions_with_published(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test listing versions includes published versions."""
    # Create and publish workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/publish",
        json={}
    )

    # List versions
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/versions"
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["versions"]) >= 1

    # Check version structure
    version = data["versions"][0]
    assert "version_id" in version
    assert "status" in version
    assert "created_at" in version


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_versions_ordering(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload, sample_workflow_spec
):
    """Test versions are listed in correct order (newest first)."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Publish multiple versions
    for i in range(3):
        if i > 0:
            # Modify draft between publishes
            modified_spec = sample_workflow_spec.copy()
            modified_spec["nodes"][0]["value"] = {"result": f"v{i+1}"}
            await authenticated_e2e_client.patch(
                f"/api/v1/workflows/{workflow_id}/draft",
                json={"spec": modified_spec}
            )

        await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish",
            json={}
        )

    # List versions
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/versions"
    )

    assert response.status_code == 200
    data = response.json()
    versions = data["versions"]

    # Verify ordering (newest first)
    for i in range(len(versions) - 1):
        v1_time = versions[i]["created_at"]
        v2_time = versions[i + 1]["created_at"]
        assert v1_time >= v2_time


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_versions_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test listing versions for non-existent workflow returns 404."""
    response = await authenticated_e2e_client.get(
        "/api/v1/workflows/nonexistent_id/versions"
    )

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_workflow_versions_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test listing versions without authentication returns 401."""
    response = await e2e_client.get("/api/v1/workflows/any_id/versions")
    assert response.status_code == 401


# =============================================================================
# Restore Workflow Version Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_restore_workflow_version_success(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload, sample_workflow_spec
):
    """Test restoring a previous workflow version."""
    # Create and publish workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/publish",
        json={}
    )

    # Get first version
    versions_response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/versions"
    )
    versions = versions_response.json()["versions"]
    version_id = versions[0]["version_id"]

    # Modify and publish new version
    modified_spec = sample_workflow_spec.copy()
    modified_spec["nodes"][0]["value"] = {"result": "modified"}
    await authenticated_e2e_client.patch(
        f"/api/v1/workflows/{workflow_id}/draft",
        json={"spec": modified_spec}
    )
    await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/publish",
        json={}
    )

    # Restore first version
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/versions/{version_id}/restore",
        json={}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_id"] == workflow_id
    # Spec should match original


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_restore_workflow_version_not_found(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test restoring non-existent version returns 404."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Try to restore non-existent version
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/versions/99999/restore",
        json={}
    )

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_restore_workflow_version_workflow_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test restoring version of non-existent workflow returns 404."""
    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/nonexistent_id/versions/1/restore",
        json={}
    )

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_restore_workflow_version_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test restoring version without authentication returns 401."""
    response = await e2e_client.post(
        "/api/v1/workflows/any_id/versions/1/restore",
        json={}
    )

    assert response.status_code == 401
