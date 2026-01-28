"""
Integration test fixtures.

Integration tests interact with real database, Valkey, and other services.
These fixtures provide setup/teardown for integration testing scenarios.
"""
from unittest.mock import AsyncMock

import pytest


# =============================================================================
# Workflow Integration Fixtures
# =============================================================================


@pytest.fixture
async def test_workflow(test_user, db_engine):
    """
    Create a test workflow in the database.

    Returns:
        Workflow: A workflow with a simple spec
    """
    from seer.database.workflow_models import Workflow  # pylint: disable=import-outside-toplevel  # Reason: Lazy import in test fixture

    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
    )
    return workflow


@pytest.fixture
async def test_workflow_run(test_workflow, db_engine):
    """
    Create a test workflow run in the database.

    Returns:
        WorkflowRun: A run for the test workflow
    """
    from seer.database.workflow_models import (  # pylint: disable=import-outside-toplevel  # Reason: Lazy import in test fixture
        WorkflowRun,
        WorkflowRunSource,
        WorkflowRunStatus,
    )

    run = await WorkflowRun.create(
        user=test_workflow.user,
        workflow=test_workflow,
        spec={"version": "2"},
        source=WorkflowRunSource.MANUAL,
        status=WorkflowRunStatus.QUEUED,
    )
    return run


@pytest.fixture
async def test_trigger_subscription(test_workflow, db_engine):
    """
    Create a test trigger subscription in the database.

    Returns:
        TriggerSubscription: A subscription for the test workflow
    """
    from seer.database.workflow_models import TriggerSubscription  # pylint: disable=import-outside-toplevel  # Reason: Lazy import in test fixture

    subscription = await TriggerSubscription.create(
        user=test_workflow.user,
        workflow=test_workflow,
        trigger_id="t1",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
    )
    return subscription


# =============================================================================
# Tool Execution Fixtures
# =============================================================================


@pytest.fixture
def mock_credential_resolver():
    """
    Mock credential resolver for testing tool execution with credentials.
    """
    resolver = AsyncMock()
    resolver.resolve.return_value = {
        "api_key": "test_key_123",
        "api_secret": "test_secret_456",
    }
    return resolver


@pytest.fixture
async def test_integration_resource(test_user):
    """
    Create a test integration resource (e.g., connected Gmail account).
    """
    from seer.database.models_integrations import IntegrationResource  # pylint: disable=import-outside-toplevel  # Reason: Lazy import in test fixture

    resource = await IntegrationResource.create(
        user=test_user,
        resource_id="res_test_123",
        provider="gmail",
        resource_type="account",
        resource_key="test@gmail.com",
        resource_metadata={"name": "Test Gmail Account"},
    )
    return resource


# =============================================================================
# Trigger Polling Fixtures
# =============================================================================


@pytest.fixture
def mock_trigger_adapter():
    """
    Mock trigger adapter for testing polling without external API calls.
    """
    adapter = AsyncMock()
    adapter.poll.return_value = [
        {
            "event_id": "evt_123",
            "data": {"message": "Test event"},
            "timestamp": "2024-01-01T00:00:00Z",
        }
    ]
    adapter.validate_config.return_value = True
    return adapter


# =============================================================================
# Worker Task Fixtures
# =============================================================================


@pytest.fixture
def mock_taskiq_broker():
    """
    Mock Taskiq broker for testing worker tasks without Valkey.
    """
    broker = AsyncMock()
    broker.startup.return_value = None
    broker.shutdown.return_value = None
    return broker


@pytest.fixture
async def test_workflow_with_run(test_workflow, db_engine):
    """
    Create a workflow with an associated run for testing execution.

    Returns:
        tuple: (workflow, run)
    """
    from seer.database.workflow_models import (  # pylint: disable=import-outside-toplevel  # Reason: Lazy import in test fixture
        WorkflowRun,
        WorkflowRunSource,
        WorkflowRunStatus,
    )

    run = await WorkflowRun.create(
        user=test_workflow.user,
        workflow=test_workflow,
        spec={"version": "2"},
        source=WorkflowRunSource.MANUAL,
        status=WorkflowRunStatus.QUEUED,
    )
    return test_workflow, run


# =============================================================================
# Stripe Integration Fixtures
# =============================================================================


@pytest.fixture
def mock_stripe_client():
    """
    Mock Stripe client for testing payment operations without hitting Stripe API.
    """
    stripe = AsyncMock()

    # Mock subscription
    stripe.Subscription.retrieve.return_value = {
        "id": "sub_test_123",
        "status": "active",
        "current_period_start": 1609459200,
        "current_period_end": 1612137600,
        "items": {
            "data": [
                {
                    "price": {
                        "id": "price_pro",
                        "nickname": "PRO",
                    }
                }
            ]
        },
    }

    # Mock customer
    stripe.Customer.retrieve.return_value = {
        "id": "cus_test_123",
        "email": "test@example.com",
        "metadata": {"user_id": "test_user_123"},
    }

    # Mock invoice
    stripe.Invoice.upcoming.return_value = {
        "amount_due": 2000,
        "currency": "usd",
    }

    return stripe
