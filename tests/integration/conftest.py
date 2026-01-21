"""
Integration test fixtures.

Integration tests interact with real database, Redis, and other services.
These fixtures provide setup/teardown for integration testing scenarios.
"""
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import pytest


# =============================================================================
# Workflow Integration Fixtures
# =============================================================================


@pytest.fixture
async def test_workflow(test_user):
    """
    Create a test workflow in the database.

    Returns:
        Workflow: A workflow with a simple spec
    """
    from seer.database.workflow_models import Workflow, WorkflowVersionStatus

    workflow = await Workflow.create(
        user=test_user,
        workflow_id="wf_test_123",
        name="Test Workflow",
        spec={
            "version": "2",
            "triggers": [
                {
                    "id": "t1",
                    "key": "test.trigger",
                    "label": "Test",
                    "config": {},
                }
            ],
            "nodes": [
                {
                    "id": "n1",
                    "type": "task",
                    "label": "Task",
                    "config": {
                        "tool_call": {
                            "tool_id": "test.tool",
                            "parameters": {},
                        }
                    },
                }
            ],
            "edges": [{"id": "e1", "source": "t1", "target": "n1"}],
        },
        status=WorkflowVersionStatus.PUBLISHED,
    )
    return workflow


@pytest.fixture
async def test_workflow_run(test_workflow):
    """
    Create a test workflow run in the database.

    Returns:
        WorkflowRun: A run for the test workflow
    """
    from seer.database.workflow_models import (
        WorkflowRun,
        WorkflowRunSource,
        WorkflowRunStatus,
    )

    run = await WorkflowRun.create(
        workflow=test_workflow,
        run_id="run_test_123",
        source=WorkflowRunSource.MANUAL,
        status=WorkflowRunStatus.PENDING,
        initial_state={},
        final_state=None,
    )
    return run


@pytest.fixture
async def test_trigger_subscription(test_workflow):
    """
    Create a test trigger subscription in the database.

    Returns:
        TriggerSubscription: A subscription for the test workflow
    """
    from seer.database.workflow_models import TriggerSubscription

    subscription = await TriggerSubscription.create(
        workflow=test_workflow,
        subscription_id="sub_test_123",
        trigger_id="t1",
        trigger_key="test.trigger",
        config={},
        last_poll_at=None,
        last_event_at=None,
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
    from seer.database.models_integrations import IntegrationResource

    resource = await IntegrationResource.create(
        user=test_user,
        resource_id="res_test_123",
        provider="gmail",
        resource_type="account",
        identifier="test@gmail.com",
        metadata={"name": "Test Gmail Account"},
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
    Mock Taskiq broker for testing worker tasks without Redis.
    """
    broker = AsyncMock()
    broker.startup.return_value = None
    broker.shutdown.return_value = None
    return broker


@pytest.fixture
async def test_workflow_with_run(test_workflow):
    """
    Create a workflow with an associated run for testing execution.

    Returns:
        tuple: (workflow, run)
    """
    from seer.database.workflow_models import (
        WorkflowRun,
        WorkflowRunSource,
        WorkflowRunStatus,
    )

    run = await WorkflowRun.create(
        workflow=test_workflow,
        run_id="run_exec_test",
        source=WorkflowRunSource.MANUAL,
        status=WorkflowRunStatus.PENDING,
        initial_state={},
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
