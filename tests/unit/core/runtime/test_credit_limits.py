from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.runtime.context import WorkflowRuntimeContext
from seer.runtime_credit_limits import check_runtime_credit_limit


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_check_runtime_credit_limit_uses_team_organization_context():
    """Runtime credit checks must pass the workflow organization for team runs."""
    mock_user = MagicMock()
    mock_org = MagicMock()
    context = WorkflowRuntimeContext(user=mock_user, organization_id=38)

    with patch("seer.database.Organization.get_or_none", new_callable=AsyncMock, return_value=mock_org) as mock_get_org, patch(
        "seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock
    ) as mock_check:
        await check_runtime_credit_limit(context, logging.getLogger(__name__))

    mock_get_org.assert_awaited_once_with(id=38)
    mock_check.assert_awaited_once_with(mock_user, mock_org)


@pytest.mark.asyncio
async def test_check_runtime_credit_limit_without_organization_uses_user_only():
    """Contexts without an org keep the existing user-scoped behavior."""
    mock_user = MagicMock()
    context = WorkflowRuntimeContext(user=mock_user)

    with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock) as mock_check:
        await check_runtime_credit_limit(context, logging.getLogger(__name__))

    mock_check.assert_awaited_once_with(mock_user, None)


@pytest.mark.asyncio
async def test_check_runtime_credit_limit_tolerates_sync_test_double_context():
    """Plain MagicMock contexts should not be treated as async org resolvers."""
    mock_user = MagicMock()
    context = MagicMock()
    context.user = mock_user

    with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock) as mock_check:
        await check_runtime_credit_limit(context, logging.getLogger(__name__))

    mock_check.assert_awaited_once_with(mock_user, context.get_organization.return_value)


@pytest.mark.asyncio
async def test_workflow_runtime_context_caches_organization_lookup():
    """Repeated credit checks should not refetch the same organization."""
    mock_org = MagicMock()
    context = WorkflowRuntimeContext(user=MagicMock(), organization_id=38)

    with patch("seer.database.Organization.get_or_none", new_callable=AsyncMock, return_value=mock_org) as mock_get_org:
        first = await context.get_organization()
        second = await context.get_organization()

    assert first is mock_org
    assert second is mock_org
    mock_get_org.assert_awaited_once_with(id=38)
