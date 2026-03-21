from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from seer.api.core.middleware.organization import OrganizationContextMiddleware


pytestmark = pytest.mark.asyncio


def _build_request(db_user):
    request = MagicMock()
    request.state = SimpleNamespace(
        db_user=db_user,
        organization=None,
        membership=None,
    )
    return request


async def test_dispatch_uses_persisted_active_org_membership():
    """Middleware should use the stored active organization when membership is valid."""
    db_user = MagicMock()
    db_user.user_id = "user_123"
    db_user.active_organization_id = 42
    db_user.save = AsyncMock()

    organization = MagicMock()
    organization.id = 42

    membership = MagicMock()
    membership.organization = organization
    membership.fetch_related = AsyncMock()

    middleware = OrganizationContextMiddleware(app=MagicMock())
    middleware._get_active_membership = AsyncMock(return_value=membership)
    middleware._get_or_create_personal_org = AsyncMock()

    request = _build_request(db_user)
    call_next = AsyncMock(return_value="response")

    response = await middleware.dispatch(request, call_next)

    assert response == "response"
    assert request.state.organization is organization
    assert request.state.membership is membership
    middleware._get_or_create_personal_org.assert_not_awaited()
    db_user.save.assert_not_awaited()


async def test_dispatch_repairs_missing_active_org_to_personal():
    """Middleware should fall back to personal org and persist the repair."""
    db_user = MagicMock()
    db_user.user_id = "user_123"
    db_user.active_organization_id = None
    db_user.save = AsyncMock()

    organization = MagicMock()
    organization.id = 7
    membership = MagicMock()

    middleware = OrganizationContextMiddleware(app=MagicMock())
    middleware._get_active_membership = AsyncMock()
    middleware._get_or_create_personal_org = AsyncMock(return_value=(organization, membership))

    request = _build_request(db_user)
    call_next = AsyncMock(return_value="response")

    response = await middleware.dispatch(request, call_next)

    assert response == "response"
    assert request.state.organization is organization
    assert request.state.membership is membership
    assert db_user.active_organization_id == organization.id
    db_user.save.assert_awaited_once_with(update_fields=["active_organization_id"])


async def test_dispatch_repairs_invalid_active_org_to_personal():
    """Middleware should repair invalid stored orgs when membership no longer exists."""
    db_user = MagicMock()
    db_user.user_id = "user_123"
    db_user.active_organization_id = 99
    db_user.save = AsyncMock()

    organization = MagicMock()
    organization.id = 5
    membership = MagicMock()

    middleware = OrganizationContextMiddleware(app=MagicMock())
    middleware._get_active_membership = AsyncMock(return_value=None)
    middleware._get_or_create_personal_org = AsyncMock(return_value=(organization, membership))

    request = _build_request(db_user)
    call_next = AsyncMock(return_value="response")

    response = await middleware.dispatch(request, call_next)

    assert response == "response"
    assert request.state.organization is organization
    assert request.state.membership is membership
    assert db_user.active_organization_id == organization.id
    db_user.save.assert_awaited_once_with(update_fields=["active_organization_id"])
