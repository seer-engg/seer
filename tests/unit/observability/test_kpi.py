"""Unit tests for KPI query functions."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.observability.kpi import (
    get_time_to_first_workflow,
    get_weekly_active_workflows,
    get_workflow_failure_rate,
)


def _mock_user(created_at=None):
    user = MagicMock()
    user.created_at = created_at or datetime(2025, 1, 1, tzinfo=timezone.utc)
    return user


@pytest.mark.asyncio
@patch("seer.observability.kpi.WorkflowRun")
async def test_weekly_active_workflows(mock_run):
    qs = MagicMock()
    mock_run.filter.return_value = qs
    qs.filter.return_value = qs
    distinct_qs = MagicMock()
    qs.distinct.return_value = distinct_qs
    distinct_qs.values_list = AsyncMock(return_value=[1, 2, 3])

    result = await get_weekly_active_workflows(user=_mock_user())
    assert result == 3


@pytest.mark.asyncio
@patch("seer.observability.kpi.WorkflowRun")
async def test_ttfw_no_runs(mock_run):
    qs = MagicMock()
    mock_run.filter.return_value = qs
    qs.order_by.return_value = qs
    qs.first = AsyncMock(return_value=None)

    result = await get_time_to_first_workflow(_mock_user())
    assert result is None


@pytest.mark.asyncio
@patch("seer.observability.kpi.WorkflowRun")
async def test_ttfw_with_runs(mock_run):
    user = _mock_user(created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    first_run = MagicMock()
    first_run.created_at = datetime(2025, 1, 1, 0, 5, 0, tzinfo=timezone.utc)  # 5 minutes later

    qs = MagicMock()
    mock_run.filter.return_value = qs
    qs.order_by.return_value = qs
    qs.first = AsyncMock(return_value=first_run)

    result = await get_time_to_first_workflow(user)
    assert result == 300.0  # 5 minutes in seconds


@pytest.mark.asyncio
@patch("seer.observability.kpi.WorkflowRun")
async def test_failure_rate_no_runs(mock_run):
    qs = MagicMock()
    mock_run.filter.return_value = qs
    qs.filter.return_value = qs
    qs.count = AsyncMock(return_value=0)

    result = await get_workflow_failure_rate(user=_mock_user())
    assert result["failure_rate"] == 0.0
    assert result["total_runs"] == 0


@pytest.mark.asyncio
@patch("seer.observability.kpi.WorkflowRun")
async def test_failure_rate_with_failures(mock_run):
    # WorkflowRun.filter(created_at__gte=cutoff)
    qs1 = MagicMock()
    mock_run.filter.return_value = qs1
    # .filter(user=user)
    qs2 = MagicMock()
    qs1.filter.return_value = qs2
    # .filter(status__in=[...]) -> terminal
    terminal_qs = MagicMock()
    qs2.filter.return_value = terminal_qs
    terminal_qs.count = AsyncMock(return_value=10)
    # terminal.filter(status=FAILED) -> failed
    failed_qs = MagicMock()
    failed_qs.count = AsyncMock(return_value=3)
    terminal_qs.filter.return_value = failed_qs

    result = await get_workflow_failure_rate(user=_mock_user())
    assert result["total_runs"] == 10
    assert result["failed_runs"] == 3
    assert result["failure_rate"] == 0.3
