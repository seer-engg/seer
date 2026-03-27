# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
"""
E2E tests for workflow export/import (clone) journey.

Tests the complete roundtrip: Create → Export → Import → Verify.
Export operates on the DRAFT version (publishing consumes the draft).
Catches serialization drift, schema mismatches, and trigger handling bugs.
"""
import pytest


pytestmark = pytest.mark.e2e


class TestWorkflowExport:
    """Tests for workflow export."""

    async def test_export_workflow(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Exporting a workflow draft should return valid export JSON."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Export Test", "spec": simple_tool_workflow_spec},
        )
        assert create_resp.status_code == 201
        workflow_id = create_resp.json()["workflow_id"]

        export_resp = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/export"
        )
        assert export_resp.status_code == 200

        data = export_resp.json()
        assert data["version"] == "1.0"
        assert "workflow" in data
        assert data["workflow"]["name"] == "Export Test"
        assert "spec" in data["workflow"]
        assert "metadata" in data
        assert data["metadata"]["original_workflow_id"] == workflow_id

    async def test_export_includes_triggers_by_default(
        self,
        authenticated_e2e_client,
        db_session,
        webhook_trigger_workflow_spec,
    ):
        """Export should include triggers when include_triggers is not set."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Trigger Export", "spec": webhook_trigger_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        export_resp = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/export"
        )
        assert export_resp.status_code == 200
        data = export_resp.json()

        assert "triggers" in data
        assert isinstance(data["triggers"], list)

    async def test_export_without_triggers(
        self,
        authenticated_e2e_client,
        db_session,
        webhook_trigger_workflow_spec,
    ):
        """Export with include_triggers=false should exclude triggers."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "No Trigger Export", "spec": webhook_trigger_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        export_resp = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/export?include_triggers=false"
        )
        assert export_resp.status_code == 200
        data = export_resp.json()

        triggers = data.get("triggers", [])
        assert triggers == [] or triggers is None


class TestWorkflowImport:
    """Tests for workflow import."""

    async def test_import_creates_new_workflow(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Importing a valid export should create a new workflow."""
        # Create and export (draft)
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Import Source", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        export_data = (await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/export"
        )).json()

        # Import as new workflow
        import_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows/import",
            json={
                "import_data": export_data,
                "name": "Imported Workflow",
            },
        )
        assert import_resp.status_code == 201

        imported = import_resp.json()
        assert imported["name"] == "Imported Workflow"
        assert imported["workflow_id"] != workflow_id
        assert "spec" in imported

    async def test_import_with_name_override(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Import should use the provided name instead of the original."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Original Name", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        export_data = (await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/export"
        )).json()

        import_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows/import",
            json={"import_data": export_data, "name": "Custom Name"},
        )
        assert import_resp.status_code == 201
        assert import_resp.json()["name"] == "Custom Name"


class TestExportImportRoundtrip:
    """Tests for full export → import → verify roundtrip."""

    async def test_roundtrip_preserves_spec(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Export → import roundtrip should preserve the workflow spec."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Roundtrip Test", "spec": simple_tool_workflow_spec},
        )
        original_id = create_resp.json()["workflow_id"]

        export_data = (await authenticated_e2e_client.get(
            f"/api/v1/workflows/{original_id}/export"
        )).json()

        import_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows/import",
            json={"import_data": export_data, "name": "Roundtrip Clone"},
        )
        cloned_id = import_resp.json()["workflow_id"]

        # Fetch both and compare specs
        original = (await authenticated_e2e_client.get(
            f"/api/v1/workflows/{original_id}"
        )).json()
        cloned = (await authenticated_e2e_client.get(
            f"/api/v1/workflows/{cloned_id}"
        )).json()

        original_nodes = {n["id"]: n for n in original["spec"].get("nodes", [])}
        cloned_nodes = {n["id"]: n for n in cloned["spec"].get("nodes", [])}

        assert set(original_nodes.keys()) == set(cloned_nodes.keys())
        for node_id in original_nodes:
            assert original_nodes[node_id]["type"] == cloned_nodes[node_id]["type"]

    async def test_cloned_workflow_can_be_published(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """A cloned workflow should be publishable."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Source", "spec": simple_tool_workflow_spec},
        )
        original_id = create_resp.json()["workflow_id"]

        export_data = (await authenticated_e2e_client.get(
            f"/api/v1/workflows/{original_id}/export"
        )).json()

        import_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows/import",
            json={"import_data": export_data, "name": "Publishable Clone"},
        )
        cloned_id = import_resp.json()["workflow_id"]

        publish_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{cloned_id}/publish"
        )
        assert publish_resp.status_code == 200

    async def test_cloned_workflow_can_be_executed(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """A cloned workflow should be executable end-to-end."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Executable Source", "spec": simple_tool_workflow_spec},
        )
        original_id = create_resp.json()["workflow_id"]

        export_data = (await authenticated_e2e_client.get(
            f"/api/v1/workflows/{original_id}/export"
        )).json()

        import_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows/import",
            json={"import_data": export_data, "name": "Runnable Clone"},
        )
        cloned_id = import_resp.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{cloned_id}/publish")

        run_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{cloned_id}/runs",
            json={"inputs": {}},
        )
        assert run_resp.status_code == 201
        assert "run_id" in run_resp.json()
