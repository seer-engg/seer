# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
"""
E2E tests for workflow lifecycle: Create -> Publish -> Execute -> Verify.

These tests use real PostgreSQL and Redis via Testcontainers to validate
the complete workflow journey from API creation to execution completion.
"""
import pytest


pytestmark = pytest.mark.e2e


class TestWorkflowCreation:
    """Tests for workflow creation via API."""

    async def test_create_workflow_returns_201(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Creating a workflow should return 201 with workflow details."""
        response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "E2E Test Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )

        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"

        data = response.json()
        assert "workflow_id" in data
        assert data["name"] == "E2E Test Workflow"
        assert "spec" in data
        assert "created_at" in data

    async def test_create_workflow_validates_spec(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Creating a workflow with invalid spec should return 400/422."""
        invalid_spec = {
            "version": "invalid",
            "nodes": [],
            "edges": [],
        }

        response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Invalid Workflow",
                "spec": invalid_spec,
            }
        )

        # Pydantic validation error returns 422
        assert response.status_code in (400, 422), f"Expected 400/422, got {response.status_code}"

    async def test_get_workflow_after_creation(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Should be able to retrieve a workflow after creating it."""
        # Create workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Retrievable Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        assert create_response.status_code == 201
        workflow_id = create_response.json()["workflow_id"]

        # Retrieve workflow
        get_response = await authenticated_e2e_client.get(f"/api/v1/workflows/{workflow_id}")

        assert get_response.status_code == 200
        data = get_response.json()
        assert data["workflow_id"] == workflow_id
        assert data["name"] == "Retrievable Workflow"


class TestWorkflowPublishing:
    """Tests for workflow publishing."""

    async def test_publish_workflow(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Publishing a workflow should succeed and update status."""
        # Create workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Publishable Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        assert create_response.status_code == 201
        workflow_id = create_response.json()["workflow_id"]

        # Publish workflow
        publish_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )

        assert publish_response.status_code == 200, f"Publish failed: {publish_response.text}"
        data = publish_response.json()
        assert data["workflow_id"] == workflow_id


class TestWorkflowExecution:
    """Tests for workflow execution."""

    async def test_execute_workflow_creates_run(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Executing a workflow should create a run and queue a task."""
        # Create and publish workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Executable Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        assert create_response.status_code == 201
        workflow_id = create_response.json()["workflow_id"]

        # Publish
        publish_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_response.status_code == 200

        # Execute workflow
        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={
                "inputs": {"test_input": "hello"},
            }
        )

        assert run_response.status_code == 201, f"Run failed: {run_response.text}"
        data = run_response.json()
        assert "run_id" in data
        assert data["workflow_id"] == workflow_id
        # Status can be queued/pending/running/succeeded depending on timing
        assert data["status"] in ("queued", "pending", "running", "succeeded")

    async def test_get_run_status(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Should be able to retrieve run status after execution."""
        # Create, publish, and run
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Status Check Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}},
        )
        run_id = run_response.json()["run_id"]

        # Get run status
        status_response = await authenticated_e2e_client.get(f"/api/v1/runs/{run_id}")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["run_id"] == run_id
        assert "status" in data
        assert "created_at" in data

    async def test_list_workflow_runs(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Should list runs for a workflow."""
        # Create and publish
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "List Runs Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]
        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Create multiple runs
        for i in range(3):
            await authenticated_e2e_client.post(
                f"/api/v1/workflows/{workflow_id}/runs",
                json={"inputs": {"iteration": i}},
            )

        # List runs
        list_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/runs"
        )

        assert list_response.status_code == 200
        data = list_response.json()
        assert "runs" in data
        assert len(data["runs"]) == 3


class TestWorkflowValidation:
    """Tests for workflow validation via publish.

    Note: There is no dedicated /validate endpoint. Validation happens
    during publish, so we test validation behavior through the publish flow.
    """

    async def test_valid_spec_publishes_successfully(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        simple_tool_workflow_spec,
    ):
        """Valid workflow specs should publish successfully."""
        # Create workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Validation Test Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        assert create_response.status_code == 201
        workflow_id = create_response.json()["workflow_id"]

        # Publish - this validates the spec
        publish_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )

        assert publish_response.status_code == 200, f"Publish failed: {publish_response.text}"

    async def test_invalid_tool_rejected_on_publish(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
    ):
        """Workflow with invalid tool should fail validation on publish."""
        invalid_spec = {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "n1",
                    "type": "tool",
                    "tool": "nonexistent.tool",
                    "inputs": {},
                },
            ],
            "edges": [],
        }

        # Create workflow (creation doesn't validate fully)
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Invalid Tool Workflow",
                "spec": invalid_spec,
            }
        )
        assert create_response.status_code == 201
        workflow_id = create_response.json()["workflow_id"]

        # Publish should fail with validation error
        publish_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )

        assert publish_response.status_code == 400, f"Expected validation error, got: {publish_response.text}"
        data = publish_response.json()
        assert "detail" in data


class TestWorkflowDeletion:
    """Tests for workflow deletion."""

    async def test_delete_workflow(
        self,
        authenticated_e2e_client,
        db_session,
        simple_tool_workflow_spec,
    ):
        """Deleting a workflow should remove it."""
        # Create workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Deletable Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        # Delete workflow
        delete_response = await authenticated_e2e_client.delete(
            f"/api/v1/workflows/{workflow_id}"
        )
        assert delete_response.status_code == 200

        # Verify it's gone
        get_response = await authenticated_e2e_client.get(f"/api/v1/workflows/{workflow_id}")
        assert get_response.status_code == 404

    async def test_delete_nonexistent_workflow_returns_404(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Deleting a non-existent workflow should return 400 or 404.

        Note: API returns 400 for invalid workflow ID formats.
        """
        response = await authenticated_e2e_client.delete(
            "/api/v1/workflows/nonexistent_workflow_id"
        )
        # API may return 400 (invalid ID format) or 404 (not found)
        assert response.status_code in (400, 404)
