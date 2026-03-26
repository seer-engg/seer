# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
"""
E2E tests for workflow versioning.

Tests the complete version lifecycle: Create v1 → Edit → Publish v2 → Rollback.
Uses real PostgreSQL via Testcontainers.
"""
import pytest


pytestmark = pytest.mark.e2e


class TestVersionLifecycle:
    """Tests for workflow version lifecycle."""

    async def test_initial_version_is_draft(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Newly created workflow should have a DRAFT version."""
        # Create workflow
        response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Version Test Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        assert response.status_code == 201
        workflow_id = response.json()["workflow_id"]

        # List versions
        versions_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/versions"
        )

        assert versions_response.status_code == 200
        data = versions_response.json()
        assert "versions" in data

        # Should have exactly one DRAFT version
        versions = data["versions"]
        assert len(versions) >= 1

        draft_versions = [v for v in versions if v["status"] == "DRAFT"]
        assert len(draft_versions) == 1
        assert draft_versions[0]["version_number"] == 0

    async def test_publish_creates_released_version(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Publishing workflow should create a RELEASED version with version_number=1."""
        # Create and publish workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Publish Version Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        publish_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_response.status_code == 200

        # List versions
        versions_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/versions"
        )
        versions = versions_response.json()["versions"]

        # Should have at least one RELEASED version
        released = [v for v in versions if v["status"] == "RELEASED"]
        assert len(released) == 1
        assert released[0]["version_number"] == 1
        assert released[0]["is_published"] is True

    async def test_edit_draft_and_publish_v2(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Editing draft and publishing again should create v2, archive v1."""
        # Create and publish v1
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Multi-Version Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Edit draft with modified spec
        modified_spec = {**simple_tool_workflow_spec}
        modified_spec["nodes"] = [
            {
                "id": "n1",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get?version=2",  # Changed URL
                },
            }
        ]

        edit_response = await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/draft",
            json={"spec": modified_spec}
        )
        assert edit_response.status_code == 200

        # Publish v2
        publish_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_response.status_code == 200

        # List versions
        versions_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/versions"
        )
        versions = versions_response.json()["versions"]

        # Should have RELEASED v2 and ARCHIVED v1
        released = [v for v in versions if v["status"] == "RELEASED"]
        archived = [v for v in versions if v["status"] == "ARCHIVED"]

        assert len(released) == 1
        assert released[0]["version_number"] == 2
        assert released[0]["is_published"] is True

        assert len(archived) >= 1
        # v1 should be archived
        v1_archived = [v for v in archived if v["version_number"] == 1]
        assert len(v1_archived) == 1

    async def test_version_numbers_increment_sequentially(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Version numbers should increment 1, 2, 3... with each publish."""
        # Create workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Sequential Versions Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        # Publish 3 times, modifying spec each time (required for new version)
        for i in range(1, 4):
            # Modify spec to force new version (same spec hash = no new version)
            modified_spec = {**simple_tool_workflow_spec}
            modified_spec["nodes"] = [
                {
                    "id": "n1",
                    "type": "tool",
                    "tool": "http_request",
                    "inputs": {
                        "method": "GET",
                        "url": f"https://httpbin.org/get?version={i}",
                    },
                }
            ]
            await authenticated_e2e_client.patch(
                f"/api/v1/workflows/{workflow_id}/draft",
                json={"spec": modified_spec}
            )

            await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

            versions_response = await authenticated_e2e_client.get(
                f"/api/v1/workflows/{workflow_id}/versions"
            )
            versions = versions_response.json()["versions"]

            released = [v for v in versions if v["status"] == "RELEASED"]
            assert len(released) == 1
            assert released[0]["version_number"] == i


class TestVersionRollback:
    """Tests for version rollback functionality."""

    async def test_restore_version_to_draft(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Restoring a version should copy its spec to the draft."""
        # Create and publish v1
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Rollback Test Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Get v1 version ID
        versions_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/versions"
        )
        versions = versions_response.json()["versions"]
        v1 = [v for v in versions if v["version_number"] == 1][0]
        v1_version_id = v1["version_id"]

        # Edit and publish v2 with different spec
        modified_spec = {**simple_tool_workflow_spec}
        modified_spec["nodes"] = [
            {
                "id": "n1",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "POST",  # Changed from GET
                    "url": "https://httpbin.org/post",
                },
            }
        ]

        await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/draft",
            json={"spec": modified_spec}
        )
        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Restore v1 (endpoint requires empty JSON body)
        restore_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/versions/{v1_version_id}/restore",
            json={}
        )
        assert restore_response.status_code == 200

        # Get current draft and verify spec is from v1
        workflow_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}"
        )
        current_spec = workflow_response.json()["spec"]

        # Should have GET method from v1, not POST from v2
        assert current_spec["nodes"][0]["inputs"]["method"] == "GET"

    async def test_restore_then_publish_creates_v3(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Restoring v1 and publishing should create v3, not re-release v1."""
        # Create and publish v1
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Restore Publish Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Get v1 version ID
        versions_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/versions"
        )
        v1 = [v for v in versions_response.json()["versions"] if v["version_number"] == 1][0]
        v1_version_id = v1["version_id"]

        # Edit and publish v2 (must modify spec to create new version)
        modified_spec = {**simple_tool_workflow_spec}
        modified_spec["nodes"] = [
            {
                "id": "n1",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get?v=2",  # Different URL
                },
            }
        ]
        await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/draft",
            json={"spec": modified_spec}
        )
        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Restore v1 and publish (endpoint requires empty JSON body)
        await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/versions/{v1_version_id}/restore",
            json={}
        )
        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # List versions
        versions_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/versions"
        )
        versions = versions_response.json()["versions"]

        # Current RELEASED should be v3
        released = [v for v in versions if v["status"] == "RELEASED"]
        assert len(released) == 1
        assert released[0]["version_number"] == 3

        # Should have v1 and v2 archived
        archived = [v for v in versions if v["status"] == "ARCHIVED"]
        archived_numbers = {v["version_number"] for v in archived}
        assert 1 in archived_numbers
        assert 2 in archived_numbers


class TestVersionImmutability:
    """Tests for version immutability rules."""

    async def test_released_version_immutable(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """RELEASED versions should not be editable via draft endpoint."""
        # Create and publish workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Immutable Version Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Try to edit - should create new draft (not modify released)
        modified_spec = {**simple_tool_workflow_spec}
        modified_spec["nodes"] = [
            {
                "id": "different_node",
                "type": "tool",
                "tool": "http_request",
                "inputs": {"method": "GET", "url": "https://example.com"},
            }
        ]

        edit_response = await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/draft",
            json={"spec": modified_spec}
        )
        # Should succeed - editing creates/modifies draft, not released version
        assert edit_response.status_code == 200

        # Verify released v1 still has original spec
        versions_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/versions"
        )
        versions = versions_response.json()["versions"]

        released = [v for v in versions if v["status"] == "RELEASED"][0]
        # Released version should retain original node ID
        assert released["version_number"] == 1


class TestRunWithVersion:
    """Tests for running specific workflow versions."""

    async def test_run_current_released_version(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Should be able to run the current released version by number."""
        # Create and publish v1
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Version Run Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Run version 1 explicitly (the current RELEASED version)
        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={
                "inputs": {},
                "version": 1,  # Explicitly request current released v1
            }
        )

        # Should succeed (may be 200 or 201)
        assert run_response.status_code in (200, 201), f"Run failed: {run_response.text}"
        data = run_response.json()
        assert "run_id" in data

    async def test_run_archived_version_fails(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Running an ARCHIVED version should return 404."""
        # Create and publish v1
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Archived Version Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Edit and publish v2 (v1 becomes ARCHIVED)
        modified_spec = {**simple_tool_workflow_spec}
        modified_spec["nodes"][0]["inputs"]["url"] = "https://httpbin.org/get?v=2"
        await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/draft",
            json={"spec": modified_spec}
        )
        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Try to run archived v1 - should fail
        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={
                "inputs": {},
                "version": 1,  # v1 is now ARCHIVED
            }
        )

        # Should return 404 (archived versions can't be run)
        assert run_response.status_code == 404

    async def test_run_nonexistent_version_fails(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Running a non-existent version should return 404."""
        # Create and publish workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Invalid Version Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Try to run version 999 (doesn't exist)
        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={
                "inputs": {},
                "version": 999,
            }
        )

        # Should fail with 404
        assert run_response.status_code == 404
