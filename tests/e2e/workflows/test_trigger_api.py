"""
E2E tests for Trigger Management API endpoints.

Tests trigger catalog, subscriptions, and trigger listening.
"""
import pytest
from httpx import AsyncClient


# =============================================================================
# Trigger Catalog Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_trigger_catalog(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test retrieving trigger catalog."""
    response = await authenticated_e2e_client.get("/api/v1/triggers")

    assert response.status_code == 200
    data = response.json()
    assert "triggers" in data
    assert isinstance(data["triggers"], list)

    # Verify trigger structure if triggers exist
    if len(data["triggers"]) > 0:
        trigger = data["triggers"][0]
        assert "key" in trigger
        assert "mode" in trigger
        assert "event_schema" in trigger


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_trigger_catalog_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test retrieving trigger catalog without authentication returns 401."""
    response = await e2e_client.get("/api/v1/triggers")
    assert response.status_code == 401


# =============================================================================
# Start Listening Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_start_listening_for_trigger(
    db_engine, authenticated_e2e_client: AsyncClient, webhook_workflow_create_payload
):
    """Test starting to listen for trigger events."""
    # Create workflow with webhook trigger
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=webhook_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Get trigger ID from spec
    trigger_id = webhook_workflow_create_payload["spec"]["triggers"][0]["id"]

    # Start listening
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/triggers/{trigger_id}/start-listening"
    )

    assert response.status_code == 200
    data = response.json()
    assert "subscription_id" in data
    assert isinstance(data["subscription_id"], int)

    # Webhook triggers should return webhook_url and secret_token
    assert "webhook_url" in data
    assert isinstance(data["webhook_url"], str)
    assert "secret_token" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_start_listening_workflow_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test start listening for non-existent workflow.

    Note: Returns 400 because the endpoint validates trigger type exists
    in the workflow spec before checking if the workflow itself exists.
    Since no workflow exists, there's no spec to check, resulting in validation error.
    """
    response = await authenticated_e2e_client.post(
        "/api/v1/workflows/nonexistent_id/triggers/t1/start-listening"
    )

    # May return 400 (validation error) or 404 (workflow not found) depending on order of checks
    assert response.status_code in [400, 404]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_start_listening_trigger_not_found(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test start listening for non-existent trigger returns 404."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]

    # Try to listen to non-existent trigger
    response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/triggers/nonexistent_trigger/start-listening"
    )

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_start_listening_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test start listening without authentication returns 401."""
    response = await e2e_client.post(
        "/api/v1/workflows/any_id/triggers/t1/start-listening"
    )
    assert response.status_code == 401


# =============================================================================
# Get Pending Events Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_pending_events_empty(
    db_engine, authenticated_e2e_client: AsyncClient, webhook_workflow_create_payload
):
    """Test getting pending events when none exist."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=webhook_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]
    trigger_id = webhook_workflow_create_payload["spec"]["triggers"][0]["id"]

    # Start listening
    await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/triggers/{trigger_id}/start-listening"
    )

    # Get pending events
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/triggers/{trigger_id}/pending-events"
    )

    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert data["events"] == []
    assert "latest_event_id" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_pending_events_with_since_parameter(
    db_engine, authenticated_e2e_client: AsyncClient, webhook_workflow_create_payload
):
    """Test getting pending events with since parameter."""
    # Create workflow
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=webhook_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]
    trigger_id = webhook_workflow_create_payload["spec"]["triggers"][0]["id"]

    # Start listening
    await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/triggers/{trigger_id}/start-listening"
    )

    # Get events since specific ID
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/triggers/{trigger_id}/pending-events?since=100"
    )

    assert response.status_code == 200
    data = response.json()
    assert "events" in data


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_pending_events_not_subscribed(
    db_engine, authenticated_e2e_client: AsyncClient, workflow_create_payload
):
    """Test getting pending events without subscription."""
    # Create workflow but don't start listening
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]
    trigger_id = workflow_create_payload["spec"]["triggers"][0]["id"]

    # Try to get events without subscription
    response = await authenticated_e2e_client.get(
        f"/api/v1/workflows/{workflow_id}/triggers/{trigger_id}/pending-events"
    )

    # Should return 404 or empty events
    assert response.status_code in [200, 404]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_pending_events_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test getting pending events without authentication returns 401."""
    response = await e2e_client.get(
        "/api/v1/workflows/any_id/triggers/t1/pending-events"
    )
    assert response.status_code == 401


# =============================================================================
# Test Trigger Subscription Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_test_trigger_subscription(
    db_engine, authenticated_e2e_client: AsyncClient, webhook_workflow_create_payload
):
    """Test testing a trigger subscription with sample event."""
    # Create workflow and start listening
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=webhook_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]
    trigger_id = webhook_workflow_create_payload["spec"]["triggers"][0]["id"]

    listen_response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/triggers/{trigger_id}/start-listening"
    )
    subscription_id = listen_response.json()["subscription_id"]

    # Test subscription with sample event matching webhook event envelope schema
    test_payload = {
        "event": {
            "id": "test_event_1",
            "trigger_key": "webhook.generic",
            "provider": "generic",
            "occurred_at": "2024-01-01T00:00:00Z",
            "data": {
                "test_field": "test_value",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
    }
    response = await authenticated_e2e_client.post(
        f"/api/v1/trigger-subscriptions/{subscription_id}/test",
        json=test_payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "inputs" in data
    # May include transformation results or validation errors
    assert isinstance(data["inputs"], dict)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_test_trigger_subscription_without_event(
    db_engine, authenticated_e2e_client: AsyncClient, webhook_workflow_create_payload
):
    """Test testing subscription without providing event data.

    webhook.generic triggers don't have a sample_event configured,
    so testing without providing event data should return 400.
    """
    # Create workflow and start listening
    create_response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=webhook_workflow_create_payload
    )
    workflow_id = create_response.json()["workflow_id"]
    trigger_id = webhook_workflow_create_payload["spec"]["triggers"][0]["id"]

    listen_response = await authenticated_e2e_client.post(
        f"/api/v1/workflows/{workflow_id}/triggers/{trigger_id}/start-listening"
    )
    subscription_id = listen_response.json()["subscription_id"]

    # Test without event data - should fail for webhook.generic
    test_payload = {}
    response = await authenticated_e2e_client.post(
        f"/api/v1/trigger-subscriptions/{subscription_id}/test",
        json=test_payload
    )

    # webhook.generic has no sample_event, so requires event payload
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    # Response format may vary - check if detail is a string or nested dict
    if isinstance(data["detail"], str):
        assert "event payload" in data["detail"].lower()
    else:
        assert "detail" in data["detail"]
        assert "event payload" in data["detail"]["detail"].lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_test_trigger_subscription_not_found(
    db_engine, authenticated_e2e_client: AsyncClient
):
    """Test testing non-existent subscription returns 404."""
    test_payload = {"event": {}}
    response = await authenticated_e2e_client.post(
        "/api/v1/trigger-subscriptions/99999/test",
        json=test_payload
    )

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_test_trigger_subscription_unauthorized(
    db_engine, e2e_client: AsyncClient
):
    """Test testing subscription without authentication returns 401."""
    test_payload = {"event": {}}
    response = await e2e_client.post(
        "/api/v1/trigger-subscriptions/1/test",
        json=test_payload
    )
    assert response.status_code == 401
