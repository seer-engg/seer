"""
End-to-end (E2E) test fixtures.

E2E tests validate complete API flows including authentication,
request validation, business logic, and database operations.
"""
from typing import AsyncGenerator

import pytest
from httpx import AsyncClient, ASGITransport


# =============================================================================
# API Client Fixtures
# =============================================================================


@pytest.fixture
async def full_app():
    """
    Create the full FastAPI application with all routers and middleware.

    This is slower than mock_app but provides complete integration testing.
    Use sparingly and prefer integration tests when possible.
    """
    import os  # pylint: disable=import-outside-toplevel  # Reason: set env before app import
    import sys  # pylint: disable=import-outside-toplevel  # Reason: need to clear module cache

    # CRITICAL: Set test environment BEFORE importing app
    os.environ["SEER_MODE"] = "self-hosted"

    # Clear modules from cache to force re-import with new env
    modules_to_clear = [
        "seer.api.main",
        "seer.config",
        "seer.api.core.middleware.auth",
    ]
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]

    # Now import the app with correct environment
    from seer.api.main import app  # pylint: disable=import-outside-toplevel  # Reason: avoid circular imports in test fixtures

    # Temporarily disable lifespan events for testing
    # This prevents database initialization conflicts
    app.router.lifespan_context = None

    return app


@pytest.fixture
async def e2e_client(full_app) -> AsyncGenerator[AsyncClient, None]:  # pylint: disable=redefined-outer-name  # Reason: pytest fixture pattern requires using fixture names as parameters
    """
    Full API client for end-to-end testing with complete middleware stack.

    Includes:
    - CORS middleware
    - Authentication middleware (mocked)
    - Usage limit middleware
    - All API routers

    Usage:
        async def test_create_workflow_e2e(e2e_client, workflow_payload):
            response = await e2e_client.post("/v1/workflows", json=workflow_payload)
            assert response.status_code == 201
    """
    transport = ASGITransport(app=full_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def authenticated_e2e_client(  # pylint: disable=redefined-outer-name  # Reason: pytest fixture pattern
    full_app, test_user
) -> AsyncGenerator[AsyncClient, None]:
    """
    Authenticated E2E client with test user injected via JWT token.

    Creates a JWT token containing the test user's information that will
    be decoded by the TokenDecodeWithoutValidationMiddleware in non-cloud mode.
    """
    import jwt  # pylint: disable=import-outside-toplevel  # Reason: avoid circular imports

    # Create a mock JWT token with test user info
    token_payload = {
        "sub": test_user.user_id,
        "email": test_user.email,
        "first_name": test_user.first_name,
        "last_name": test_user.last_name,
    }

    # Encode without signature (middleware decodes without verification)
    token = jwt.encode(token_payload, "test_secret", algorithm="HS256")

    transport = ASGITransport(app=full_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Add Authorization header with bearer token
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


# =============================================================================
# Workflow API Fixtures
# =============================================================================


@pytest.fixture
def workflow_create_payload(sample_workflow_spec):
    """
    Valid payload for creating a workflow via API.
    """
    return {
        "name": "Test Workflow",
        "description": "Created via API test",
        "spec": sample_workflow_spec,
    }


@pytest.fixture
def workflow_update_payload():
    """
    Valid payload for updating a workflow via API.
    """
    return {
        "name": "Updated Workflow Name",
        "description": "Updated description",
    }


@pytest.fixture
def webhook_workflow_create_payload():
    """
    Workflow payload with a webhook trigger for start-listening tests.

    Uses webhook.generic trigger type which supports the start-listening endpoint.
    The start_listening_for_trigger endpoint only works with webhook triggers
    (keys starting with "webhook."), not polling triggers.
    """
    return {
        "name": "Webhook Test Workflow",
        "description": "Created via API test with webhook trigger",
        "spec": {
            "version": "2",
            "triggers": [
                {
                    "id": "t1",
                    "key": "webhook.generic",
                    "mode": "webhook",
                    "provider_config": {},  # webhook.generic requires empty provider_config
                }
            ],
            "nodes": [
                {
                    "id": "n1",
                    "type": "agent",
                    "inputs": {
                        "model": "test-model",
                        "prompt": "Process the trigger data",
                    },
                    "outputs": {"mode": "text"},
                }
            ],
            "edges": [
                {
                    "source": "t1",
                    "target": "n1",
                    "type": "trigger",
                }
            ],
        }
    }


@pytest.fixture
def simple_workflow_create_payload():
    """
    Simple workflow payload without triggers for execution tests.

    Workflows without triggers can be executed directly without
    needing trigger subscription setup.
    """
    return {
        "name": "Simple Test Workflow",
        "description": "Created via API test without triggers",
        "spec": {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "n1",
                    "type": "agent",
                    "inputs": {
                        "model": "test-model",
                        "prompt": "Process the input data",
                    },
                    "outputs": {"mode": "text"},
                }
            ],
            "edges": [],
        }
    }


# =============================================================================
# Execution API Fixtures
# =============================================================================


@pytest.fixture
def workflow_run_payload():
    """
    Valid payload for executing a workflow via API.
    """
    return {
        "input": {
            "param1": "value1",
            "param2": 42,
        }
    }


# =============================================================================
# Trigger API Fixtures
# =============================================================================


@pytest.fixture
def trigger_subscription_payload():
    """
    Valid payload for creating a trigger subscription via API.
    """
    return {
        "trigger_id": "t1",
        "config": {
            "query": "test",
            "interval_seconds": 300,
        },
    }


# =============================================================================
# Tool API Fixtures
# =============================================================================


@pytest.fixture
def tool_search_query():
    """
    Sample tool search query for testing tool discovery API.
    """
    return {
        "query": "send email",
        "limit": 10,
    }


# =============================================================================
# Usage & Billing API Fixtures
# =============================================================================


@pytest.fixture
def stripe_webhook_payload():
    """
    Mock Stripe webhook payload for testing webhook handlers.
    """
    return {
        "id": "evt_test_123",
        "object": "event",
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_test_123",
                "customer": "cus_test_123",
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
        },
    }
