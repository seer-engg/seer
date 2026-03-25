# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
"""
E2E tests for trigger-based workflow execution.

Tests webhook triggers, polling triggers, and trigger subscription management.
"""
import pytest


pytestmark = pytest.mark.e2e


class TestWebhookTriggers:
    """Tests for webhook trigger execution."""

    async def test_create_workflow_with_webhook_trigger(
        self,
        authenticated_e2e_client,
        db_session,
        webhook_trigger_workflow_spec,
    ):
        """Creating a workflow with webhook trigger should succeed."""
        response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Webhook Workflow",
                "spec": webhook_trigger_workflow_spec,
            }
        )

        assert response.status_code == 201, f"Create failed: {response.text}"
        data = response.json()
        assert "workflow_id" in data

        # Verify trigger is in spec
        spec = data["spec"]
        assert len(spec.get("triggers", [])) == 1
        assert spec["triggers"][0]["key"] == "webhook.generic"

    @pytest.mark.skip(reason="Trigger provisioning requires external resources - times out in E2E tests")
    async def test_start_listening_for_webhook_trigger(
        self,
        authenticated_e2e_client,
        db_session,
        webhook_trigger_workflow_spec,
    ):
        """Should be able to start listening for webhook events."""
        # Create and publish workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Webhook Listener Workflow",
                "spec": webhook_trigger_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]

        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # Start listening - this creates a trigger subscription
        listen_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/triggers/webhook/start-listening",
            json={"trigger_id": "webhook"},
        )

        # May be 200 or 201 depending on whether subscription already exists
        assert listen_response.status_code in (200, 201, 400), f"Start listening failed: {listen_response.text}"


class TestTriggerSubscriptions:
    """Tests for trigger subscription management."""

    @pytest.mark.skip(reason="Trigger subscription listing requires trigger provisioning - times out")
    async def test_list_trigger_subscriptions(
        self,
        authenticated_e2e_client,
        db_session,
        webhook_trigger_workflow_spec,
    ):
        """Should list trigger subscriptions for a workflow."""
        # Create and publish workflow
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Subscription List Workflow",
                "spec": webhook_trigger_workflow_spec,
            }
        )
        workflow_id = create_response.json()["workflow_id"]
        await authenticated_e2e_client.post(f"/api/v1/workflows/{workflow_id}/publish")

        # List subscriptions
        list_response = await authenticated_e2e_client.get(
            f"/api/v1/workflows/{workflow_id}/trigger-subscriptions"
        )

        assert list_response.status_code == 200, f"List failed: {list_response.text}"
        data = list_response.json()
        # Should have subscriptions or items key
        assert "subscriptions" in data or "items" in data


class TestTriggerCatalog:
    """Tests for trigger catalog/discovery."""

    async def test_get_trigger_catalog(
        self,
        authenticated_e2e_client,
        db_session,
    ):
        """Should return list of available triggers."""
        # Endpoint is /api/v1/triggers (not /triggers/catalog)
        response = await authenticated_e2e_client.get("/api/v1/triggers")

        assert response.status_code == 200, f"Catalog failed: {response.text}"
        data = response.json()

        # Should have list of triggers
        assert "triggers" in data
        triggers = data["triggers"]
        assert isinstance(triggers, list)

        # Should include common trigger types
        trigger_keys = [t.get("key", t.get("id")) for t in triggers]
        assert len(trigger_keys) > 0, "No triggers in catalog"


class TestManualTriggerExecution:
    """Tests for manually triggering workflow execution with custom event data."""

    async def test_execute_with_custom_trigger_event(
        self,
        authenticated_e2e_client,
        db_session,
        e2e_checkpointer,
        taskiq_direct_executor,
        webhook_trigger_workflow_spec,
    ):
        """Should execute workflow with custom trigger event data."""
        # Create and publish
        create_response = await authenticated_e2e_client.post(
            "/api/v1/workflows",
            json={
                "name": "Custom Trigger Workflow",
                "spec": webhook_trigger_workflow_spec,
            }
        )
        assert create_response.status_code == 201
        workflow_id = create_response.json()["workflow_id"]

        publish_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/publish"
        )
        assert publish_response.status_code == 200, f"Publish failed: {publish_response.text}"

        # Execute with custom trigger event
        run_response = await authenticated_e2e_client.post(
            f"/api/v1/workflows/{workflow_id}/runs",
            json={
                "inputs": {},
                "trigger_event_override": {
                    "trigger_key": "webhook.generic",
                    "data": {
                        "custom_field": "test_value",
                        "nested": {"key": "value"},
                    },
                },
                "trigger_id": "webhook",
            }
        )

        assert run_response.status_code == 201, f"Run failed: {run_response.text}"
        data = run_response.json()
        assert "run_id" in data
