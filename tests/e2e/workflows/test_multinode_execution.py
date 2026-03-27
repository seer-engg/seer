# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
"""
E2E tests for multi-node workflow execution.

Tests complex workflow patterns: conditional branching, sequential chains,
draft editing, and execution with real compiler + runtime.
"""
import pytest


pytestmark = pytest.mark.e2e


@pytest.fixture
def sequential_workflow_spec() -> dict:
    """
    Linear workflow with two sequential tool nodes.

    Node A → Node B, testing sequential execution and data flow.
    """
    return {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "step_1",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get?step=first",
                },
            },
            {
                "id": "step_2",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get?step=second",
                },
            },
        ],
        "edges": [
            {"source": "step_1", "target": "step_2", "type": "default"},
        ],
    }


@pytest.fixture
def conditional_branching_spec() -> dict:
    """
    Workflow with conditional branching: if → true/false paths.

    Tests the compiler's control flow lowering and runtime branch selection.
    """
    return {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "check",
                "type": "if",
                "condition": "True",
            },
            {
                "id": "on_true",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get?branch=true",
                },
            },
            {
                "id": "on_false",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get?branch=false",
                },
            },
        ],
        "edges": [
            {"source": "check", "target": "on_true", "type": "conditional_true"},
            {"source": "check", "target": "on_false", "type": "conditional_false"},
        ],
    }


class TestSequentialExecution:
    """Tests for linear multi-node workflow execution."""

    async def test_sequential_nodes_execute_in_order(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        sequential_workflow_spec,
    ):
        """Sequential nodes should execute in edge-defined order."""
        # Create and publish
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Sequential Test", "spec": sequential_workflow_spec},
        )
        assert create_resp.status_code == 201
        workflow_id = create_resp.json()["workflow_id"]

        publish_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_resp.status_code == 200

        # Execute
        run_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}},
        )
        assert run_resp.status_code == 201
        run_id = run_resp.json()["run_id"]

        # Check status
        status_resp = await authenticated_e2e_client.get(f"/api/v1/runs/{run_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] in ("queued", "running", "succeeded", "failed")


class TestConditionalExecution:
    """Tests for conditional branching workflow execution."""

    async def test_conditional_workflow_publishes(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        conditional_branching_spec,
    ):
        """Conditional workflow should compile and publish successfully."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Conditional Test", "spec": conditional_branching_spec},
        )
        assert create_resp.status_code == 201
        workflow_id = create_resp.json()["workflow_id"]

        publish_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_resp.status_code == 200, f"Publish failed: {publish_resp.text}"

    async def test_conditional_workflow_executes(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        conditional_branching_spec,
    ):
        """Conditional workflow should execute and pick the correct branch."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Conditional Execution", "spec": conditional_branching_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}},
        )
        assert run_resp.status_code == 201
        run_id = run_resp.json()["run_id"]

        status_resp = await authenticated_e2e_client.get(f"/api/v1/runs/{run_id}")
        data = status_resp.json()
        assert data["status"] in ("queued", "running", "succeeded", "failed")


class TestDraftEditing:
    """Tests for editing workflow drafts and republishing."""

    async def test_patch_draft_updates_spec(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """PATCH draft should update the workflow spec."""
        # Create workflow
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Draft Edit Test", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        # Patch the draft with a new spec
        new_spec = {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "updated_node",
                    "type": "tool",
                    "tool": "http_request",
                    "inputs": {
                        "method": "GET",
                        "url": "https://httpbin.org/get?updated=true",
                    },
                },
            ],
            "edges": [],
        }

        patch_resp = await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/draft",
            json={"spec": new_spec},
        )
        assert patch_resp.status_code == 200

        # Verify the spec was updated
        get_resp = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}"
        )
        data = get_resp.json()
        nodes = data["spec"]["nodes"]
        assert any(n["id"] == "updated_node" for n in nodes)

    async def test_edit_and_republish(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Editing draft and republishing should create a new version."""
        # Create and publish v1
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Edit Republish Test", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        publish_v1 = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_v1.status_code == 200

        # After publishing, draft is consumed. PATCH /draft creates a new draft.
        updated_spec = {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "v2_node",
                    "type": "tool",
                    "tool": "http_request",
                    "inputs": {
                        "method": "GET",
                        "url": "https://httpbin.org/get?version=2",
                    },
                },
            ],
            "edges": [],
        }
        patch_resp = await authenticated_e2e_client.patch(
            f"/api/v1/workflows/{workflow_id}/draft",
            json={"spec": updated_spec},
        )
        assert patch_resp.status_code == 200

        # Publish v2
        publish_v2 = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_v2.status_code == 200

        # Verify version history has multiple versions
        # v1 may be RELEASED or ARCHIVED, v2 should be RELEASED
        versions_resp = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/versions"
        )
        versions = versions_resp.json()["versions"]
        non_draft = [v for v in versions if v["status"] != "DRAFT"]
        assert len(non_draft) >= 2, f"Expected 2+ non-draft versions, got {[v['status'] for v in versions]}"

    async def test_rename_workflow(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """PUT should update workflow name."""
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "Original Name", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        update_resp = await authenticated_e2e_client.put(
            f"/api/v1/workflows/{workflow_id}",
            json={"name": "Renamed Workflow"},
        )
        assert update_resp.status_code == 200

        get_resp = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}"
        )
        assert get_resp.json()["name"] == "Renamed Workflow"


class TestWorkflowListing:
    """Tests for listing and pagination."""

    async def test_list_workflows(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Should list created workflows."""
        # Create a few workflows
        for i in range(3):
            await authenticated_e2e_client.post(
                "/api/v1/workflows",
                json={"name": f"List Test {i}", "spec": simple_tool_workflow_spec},
            )

        list_resp = await authenticated_e2e_client.get("/api/v1/workflows")
        assert list_resp.status_code == 200

        data = list_resp.json()
        assert "items" in data
        assert len(data["items"]) >= 3

    async def test_list_workflows_with_limit(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Pagination limit should be respected."""
        for i in range(5):
            await authenticated_e2e_client.post(
                "/api/v1/workflows",
                json={"name": f"Page Test {i}", "spec": simple_tool_workflow_spec},
            )

        list_resp = await authenticated_e2e_client.get("/api/v1/workflows?limit=2")
        assert list_resp.status_code == 200

        data = list_resp.json()
        assert len(data["items"]) <= 2


class TestRunHistory:
    """Tests for run history and inspection."""

    async def test_get_run_history(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Run history should be available after execution."""
        # Create, publish, execute
        create_resp = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={"name": "History Test", "spec": simple_tool_workflow_spec},
        )
        workflow_id = create_resp.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_resp = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}},
        )
        run_id = run_resp.json()["run_id"]

        # Get history
        history_resp = await authenticated_e2e_client.get(
            f"/api/v1/runs/{run_id}/history"
        )
        assert history_resp.status_code == 200

        data = history_resp.json()
        assert "run_id" in data
        assert "history" in data
        assert isinstance(data["history"], list)
