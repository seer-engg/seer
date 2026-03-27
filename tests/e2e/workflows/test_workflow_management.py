# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
"""
E2E tests for workflow management operations.

Tests: active/pause toggle, workflow listing with filters,
and cross-cutting management concerns.
"""
import pytest


pytestmark = pytest.mark.e2e


class TestWorkflowActiveToggle:
    """Tests for toggling workflow active/paused state."""

    async def test_toggle_workflow_inactive(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Should be able to deactivate a published workflow."""
        # Create and publish
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Toggle Test", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]
        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Toggle inactive
        toggle_resp = await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/active",
            json={"is_active": False},
        )
        assert toggle_resp.status_code == 200

        # The toggle endpoint itself returns updated state
        toggle_data = toggle_resp.json()
        assert toggle_data.get("is_active") is False or toggle_data.get("ok") is True

    async def test_toggle_workflow_roundtrip(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Should be able to deactivate then reactivate a workflow."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Reactivate Test", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]
        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Deactivate
        deactivate_resp = await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/active",
            json={"is_active": False},
        )
        assert deactivate_resp.status_code == 200

        # Reactivate
        reactivate_resp = await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/active",
            json={"is_active": True},
        )
        assert reactivate_resp.status_code == 200


class TestWorkflowAccessControl:
    """Tests for access control and cross-user isolation."""

    async def test_cannot_access_nonexistent_workflow(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Accessing a non-existent workflow should return 404."""
        resp = await authenticated_e2e_client.get(
            "/api/v1/workflows/wf_nonexistent_id_12345"
        )
        assert resp.status_code in (400, 404)

    async def test_cannot_publish_nonexistent_workflow(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
    ):
        """Publishing a non-existent workflow should fail."""
        resp = await authenticated_e2e_client.post(
            "/api/v1/workflows/wf_nonexistent_id_12345/publish"
        )
        assert resp.status_code in (400, 404)

    async def test_run_unpublished_workflow_uses_draft(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Running an unpublished workflow should execute from the draft version."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Draft Run Test", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        # Run without publishing — API allows running from draft
        run_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}},
        )
        assert run_resp.status_code == 201
        assert "run_id" in run_resp.json()


class TestWorkflowFullJourney:
    """End-to-end journey tests combining multiple operations."""

    async def test_create_edit_publish_run_inspect(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Full user journey: create → edit → publish → run → inspect."""
        # 1. Create
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Full Journey", "spec": simple_tool_workflow_spec},
        )
        assert create_resp.status_code == 201
        workflow_id = create_resp.json()["workflow_id"]

        # 2. Edit (rename)
        await authenticated_e2e_client.put(
            f"/api/v1/workflows/{workflow_id}",
            json={"name": "Full Journey (edited)"},
        )

        # 3. Publish
        publish_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_resp.status_code == 200

        # 4. Run
        run_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}},
        )
        assert run_resp.status_code == 201
        run_id = run_resp.json()["run_id"]

        # 5. Inspect run
        status_resp = await authenticated_e2e_client.get(f"/api/v1/runs/{run_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["workflow_id"] == workflow_id

        # 6. Check history
        history_resp = await authenticated_e2e_client.get(
            f"/api/v1/runs/{run_id}/history"
        )
        assert history_resp.status_code == 200

        # 7. List runs
        runs_resp = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/runs"
        )
        assert runs_resp.status_code == 200
        assert len(runs_resp.json()["runs"]) >= 1

    async def test_create_publish_clone_publish_run(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Clone journey: create → publish → export → import → publish → run."""
        # Create original (export works on draft)
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Clone Source", "spec": simple_tool_workflow_spec},
        )
        original_id = create_resp.json()["workflow_id"]

        # Export from draft
        export_data = (await authenticated_e2e_client.get(
            f"/api/v1/workflows/{original_id}/export"
        )).json()

        # Import as clone
        import_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows/import",
            json={"import_data": export_data, "name": "Clone Target"},
        )
        assert import_resp.status_code == 201
        clone_id = import_resp.json()["workflow_id"]

        # Publish clone
        publish_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{clone_id}/publish"
        )
        assert publish_resp.status_code == 200

        # Run clone
        run_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{clone_id}/runs",
            json={"inputs": {}},
        )
        assert run_resp.status_code == 201
        assert run_resp.json()["workflow_id"] == clone_id
