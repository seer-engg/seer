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
    # Import here to avoid circular imports
    from seer.api.main import app

    # Temporarily disable lifespan events for testing
    # This prevents database initialization conflicts
    app.router.lifespan_context = None

    return app


@pytest.fixture
async def e2e_client(full_app) -> AsyncGenerator[AsyncClient, None]:
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
async def authenticated_e2e_client(full_app, test_user) -> AsyncGenerator[AsyncClient, None]:
    """
    Authenticated E2E client with test user injected in middleware.

    Bypasses Clerk authentication for testing protected endpoints.
    """
    # Mock authentication to inject test_user
    from unittest.mock import patch

    with patch("seer.api.core.middleware.auth.get_current_user", return_value=test_user):
        transport = ASGITransport(app=full_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Add auth header (content doesn't matter since we're mocking)
            client.headers["Authorization"] = "Bearer test_token_123"
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
