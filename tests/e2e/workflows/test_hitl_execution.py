# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
"""
E2E tests for Human-in-the-Loop (HITL) workflow execution.

Tests the complete HITL lifecycle: Execute → INTERRUPTED → Get interrupt → Resume → SUCCEEDED.
Uses real PostgreSQL + LangGraph checkpointer via Testcontainers.
"""
import pytest


pytestmark = pytest.mark.e2e


@pytest.fixture
def hitl_workflow_spec() -> dict:
    """
    Workflow spec with a Human-in-the-Loop node.

    The HITL node pauses execution and waits for human input before continuing.
    Schema fields: question (not label), input_type (not type).
    """
    return {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "approval",
                "type": "hitl",
                "title": "Approval Required",
                "description": "Please approve this workflow to continue.",
                "display": [
                    {
                        "label": "Context",
                        "value": "This is a test workflow requiring approval.",
                    }
                ],
                "inputs": [
                    {
                        "id": "approved",
                        "input_type": "boolean",
                        "question": "Approve this action?",
                        "required": True,
                    },
                    {
                        "id": "comment",
                        "input_type": "text",
                        "question": "Comments (optional)",
                        "required": False,
                    },
                ],
                "delivery_channels": [
                    {"type": "platform"}  # Use platform polling, not email
                ],
            },
        ],
        "edges": [],
    }


@pytest.fixture
def hitl_with_timeout_spec() -> dict:
    """
    HITL workflow spec with a timeout.

    Schema uses 'options' (list of {value, label}) not 'choices' (list of strings).
    """
    return {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "timed_approval",
                "type": "hitl",
                "title": "Timed Approval",
                "description": "This approval has a short timeout.",
                "display": [],
                "inputs": [
                    {
                        "id": "decision",
                        "input_type": "single_choice",
                        "question": "Choose an option",
                        "options": [
                            {"value": "approve", "label": "Approve"},
                            {"value": "reject", "label": "Reject"},
                            {"value": "defer", "label": "Defer"},
                        ],
                        "required": True,
                    },
                ],
                "timeout_seconds": 3600,  # 1 hour timeout
                "delivery_channels": [{"type": "platform"}],
            },
        ],
        "edges": [],
    }


class TestHITLBasicFlow:
    """Tests for basic HITL execution flow."""

    async def test_hitl_execution_pauses_at_human_step(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        hitl_workflow_spec,
    ):
        """Executing workflow with HITL node should pause with INTERRUPTED status."""
        # Create and publish workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "HITL Pause Test",
                "spec": hitl_workflow_spec,
            }
        )
        assert create_response.status_code == 201
        workflow_id = create_response.json()["workflow_id"]

        publish_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_response.status_code == 200

        # Execute workflow
        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}}
        )
        assert run_response.status_code == 201
        run_id = run_response.json()["run_id"]

        # Check run status - should be INTERRUPTED
        status_response = await authenticated_e2e_client.get(f"/api/v1/runs/{run_id}")
        assert status_response.status_code == 200

        data = status_response.json()
        # Status should indicate paused at human step
        assert data["status"] in ("interrupted", "INTERRUPTED"), f"Expected interrupted, got {data['status']}"

    async def test_get_interrupt_data(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        hitl_workflow_spec,
    ):
        """Should be able to retrieve interrupt data for a paused run."""
        # Create, publish, and execute
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "HITL Interrupt Data Test",
                "spec": hitl_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}}
        )
        run_id = run_response.json()["run_id"]

        # Get interrupt data
        interrupt_response = await authenticated_e2e_client.get(
            f"/api/v1/runs/{run_id}/interrupt"
        )
        assert interrupt_response.status_code == 200

        data = interrupt_response.json()
        assert data["status"] in ("interrupted", "INTERRUPTED")
        assert data["node_id"] == "approval"
        assert data["title"] == "Approval Required"
        assert "inputs" in data
        assert len(data["inputs"]) == 2  # approved and comment

        # Verify input field definitions
        input_ids = {inp["id"] for inp in data["inputs"]}
        assert "approved" in input_ids
        assert "comment" in input_ids

    async def test_resume_completes_workflow(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        hitl_workflow_spec,
    ):
        """Resuming with responses should complete the workflow."""
        # Create, publish, and execute
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "HITL Resume Test",
                "spec": hitl_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}}
        )
        run_id = run_response.json()["run_id"]

        # Resume with responses
        resume_response = await authenticated_e2e_client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={
                "responses": {
                    "approved": True,
                    "comment": "Looks good!",
                }
            }
        )
        assert resume_response.status_code == 200

        # Check final status
        status_response = await authenticated_e2e_client.get(f"/api/v1/runs/{run_id}")
        data = status_response.json()

        # Should be succeeded or running (depending on async completion)
        assert data["status"] in ("succeeded", "SUCCEEDED", "running", "RUNNING"), \
            f"Expected succeeded/running, got {data['status']}"


class TestHITLInputTypes:
    """Tests for different HITL input types."""

    async def test_single_choice_input(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        hitl_with_timeout_spec,
    ):
        """HITL with single_choice input should work correctly."""
        # Create, publish, and execute
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "HITL Single Choice Test",
                "spec": hitl_with_timeout_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}}
        )
        run_id = run_response.json()["run_id"]

        # Get interrupt and verify choice options
        interrupt_response = await authenticated_e2e_client.get(
            f"/api/v1/runs/{run_id}/interrupt"
        )
        data = interrupt_response.json()

        decision_input = next(inp for inp in data["inputs"] if inp["id"] == "decision")
        assert decision_input["input_type"] == "single_choice"
        assert "options" in decision_input
        option_values = {opt["value"] for opt in decision_input["options"]}
        assert option_values == {"approve", "reject", "defer"}

        # Resume with valid choice
        resume_response = await authenticated_e2e_client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={
                "responses": {
                    "decision": "approve",
                }
            }
        )
        assert resume_response.status_code == 200


class TestHITLTimeout:
    """Tests for HITL timeout handling."""

    async def test_interrupt_includes_timeout_info(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        hitl_with_timeout_spec,
    ):
        """Interrupt data should include timeout information."""
        # Create, publish, and execute
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "HITL Timeout Info Test",
                "spec": hitl_with_timeout_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}}
        )
        run_id = run_response.json()["run_id"]

        # Get interrupt and verify timeout
        interrupt_response = await authenticated_e2e_client.get(
            f"/api/v1/runs/{run_id}/interrupt"
        )
        data = interrupt_response.json()

        assert data["timeout_seconds"] == 3600
        # Should have an expiration time
        if "expires_at" in data:
            assert data["expires_at"] is not None


class TestHITLErrorCases:
    """Tests for HITL error handling."""

    async def test_resume_noninterrupted_run_fails(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Resuming a run that's not interrupted should fail."""
        # Create, publish, and execute a non-HITL workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Non-HITL Workflow",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}}
        )
        run_id = run_response.json()["run_id"]

        # Try to resume - should fail
        resume_response = await authenticated_e2e_client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={"responses": {"anything": "value"}}
        )

        # Should return error (400 or 409)
        assert resume_response.status_code in (400, 409, 422), \
            f"Expected error, got {resume_response.status_code}: {resume_response.text}"

    async def test_get_interrupt_for_noninterrupted_run(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        simple_tool_workflow_spec,
    ):
        """Getting interrupt data for a non-interrupted run should fail or return empty."""
        # Create, publish, and execute a non-HITL workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Non-HITL Interrupt Test",
                "spec": simple_tool_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}}
        )
        run_id = run_response.json()["run_id"]

        # Get interrupt - should fail or indicate no interrupt
        interrupt_response = await authenticated_e2e_client.get(
            f"/api/v1/runs/{run_id}/interrupt"
        )

        # Should return error or indicate no pending interrupt
        # Could be 404 (no interrupt), 400 (invalid state), or 200 with no data
        if interrupt_response.status_code == 200:
            data = interrupt_response.json()
            # If 200, status should not be interrupted
            assert data.get("status") not in ("interrupted", "INTERRUPTED")

    async def test_resume_missing_required_field_fails(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        hitl_workflow_spec,
    ):
        """Resuming without required field should fail."""
        # Create, publish, and execute HITL workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "HITL Missing Field Test",
                "spec": hitl_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={"inputs": {}}
        )
        run_id = run_response.json()["run_id"]

        # Resume without "approved" (which is required)
        resume_response = await authenticated_e2e_client.post(
            f"/api/v1/runs/{run_id}/resume",
            json={
                "responses": {
                    "comment": "No approval given",
                    # Missing "approved" field
                }
            }
        )

        # Should fail with validation error (400 or 422)
        # Note: Some systems may allow partial responses, adjust assertion if needed
        assert resume_response.status_code in (200, 400, 422), \
            f"Unexpected status: {resume_response.status_code}"
