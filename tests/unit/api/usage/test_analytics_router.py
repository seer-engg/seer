"""
Unit tests for analytics endpoints in seer.api.usage.router.
"""
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


PERIOD_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
PERIOD_END = datetime(2025, 2, 1, tzinfo=timezone.utc)


@pytest.fixture
def mock_request(mock_user):
    """Create a mock request with authenticated user."""
    request = MagicMock()
    request.state.db_user = mock_user
    return request


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetUsageAnalytics:
    """Tests for GET /api/usage/analytics endpoint."""

    async def test_full_response_shape(self, mock_request):
        """Test the full analytics overview response shape."""
        from seer.api.usage.router import get_usage_analytics

        by_model = [
            {
                "model": "gpt-4o",
                "provider": "openai",
                "total_cost": Decimal("1.50"),
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
                "total_tokens": 1500,
                "call_count": 3,
            },
        ]
        by_operation = [
            {
                "operation": "workflow_execution",
                "total_cost": Decimal("1.50"),
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
                "total_tokens": 1500,
                "call_count": 3,
            },
        ]
        daily_trend = [
            {"date": "2025-01-15", "total_cost": Decimal("1.50"), "total_tokens": 1500, "call_count": 3},
        ]

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_by_model", new_callable=AsyncMock, return_value=by_model),
            patch("seer.api.usage.router.get_llm_usage_by_operation", new_callable=AsyncMock, return_value=by_operation),
            patch("seer.api.usage.router.get_llm_usage_daily_trend", new_callable=AsyncMock, return_value=daily_trend),
        ):
            result = await get_usage_analytics(
                request=mock_request,
                start=None,
                end=None,
                model=None,
                operation=None,
            )

        assert result.period_start == PERIOD_START
        assert result.period_end == PERIOD_END
        assert result.total_cost == 1.50
        assert result.total_tokens == 1500
        assert result.total_calls == 3
        assert len(result.by_model) == 1
        assert result.by_model[0].model == "gpt-4o"
        assert len(result.by_operation) == 1
        assert result.by_operation[0].operation == "workflow_execution"
        assert len(result.daily_trend) == 1
        assert result.daily_trend[0].date == "2025-01-15"

    async def test_custom_period(self, mock_request):
        """Test analytics with custom time period."""
        from seer.api.usage.router import get_usage_analytics

        custom_start = datetime(2025, 1, 10, tzinfo=timezone.utc)
        custom_end = datetime(2025, 1, 20, tzinfo=timezone.utc)

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(custom_start, custom_end)) as mock_resolve,
            patch("seer.api.usage.router.get_llm_usage_by_model", new_callable=AsyncMock, return_value=[]),
            patch("seer.api.usage.router.get_llm_usage_by_operation", new_callable=AsyncMock, return_value=[]),
            patch("seer.api.usage.router.get_llm_usage_daily_trend", new_callable=AsyncMock, return_value=[]),
        ):
            result = await get_usage_analytics(
                request=mock_request,
                start=custom_start,
                end=custom_end,
                model=None,
                operation=None,
            )

        assert result.period_start == custom_start
        assert result.period_end == custom_end
        mock_resolve.assert_called_once()

    async def test_filter_params_passed(self, mock_request):
        """Test that model and operation filters are passed to query functions."""
        from seer.api.usage.router import get_usage_analytics

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_by_model", new_callable=AsyncMock, return_value=[]) as mock_by_model,
            patch("seer.api.usage.router.get_llm_usage_by_operation", new_callable=AsyncMock, return_value=[]) as mock_by_op,
            patch("seer.api.usage.router.get_llm_usage_daily_trend", new_callable=AsyncMock, return_value=[]) as mock_daily,
        ):
            await get_usage_analytics(
                request=mock_request,
                start=None,
                end=None,
                model="gpt-4o",
                operation="chat_message",
            )

        # Verify filters were passed through
        for mock_fn in [mock_by_model, mock_by_op, mock_daily]:
            call_kwargs = mock_fn.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["operation"] == "chat_message"

    async def test_empty_results(self, mock_request):
        """Test analytics with no data."""
        from seer.api.usage.router import get_usage_analytics

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_by_model", new_callable=AsyncMock, return_value=[]),
            patch("seer.api.usage.router.get_llm_usage_by_operation", new_callable=AsyncMock, return_value=[]),
            patch("seer.api.usage.router.get_llm_usage_daily_trend", new_callable=AsyncMock, return_value=[]),
        ):
            result = await get_usage_analytics(
                request=mock_request,
                start=None,
                end=None,
                model=None,
                operation=None,
            )

        assert result.total_cost == 0.0
        assert result.total_tokens == 0
        assert result.total_calls == 0
        assert result.by_model == []
        assert result.by_operation == []
        assert result.daily_trend == []

    async def test_unauthenticated_request_raises_401(self):
        """Test that unauthenticated request raises 401."""
        from fastapi import HTTPException
        from seer.api.usage.router import get_usage_analytics

        request = MagicMock()
        request.state.db_user = None

        with pytest.raises(HTTPException) as exc_info:
            await get_usage_analytics(
                request=request,
                start=None,
                end=None,
                model=None,
                operation=None,
            )
        assert exc_info.value.status_code == 401

    async def test_totals_computed_from_by_model(self, mock_request):
        """Test that totals are summed from by_model results."""
        from seer.api.usage.router import get_usage_analytics

        by_model = [
            {
                "model": "gpt-4o",
                "provider": "openai",
                "total_cost": Decimal("1.00"),
                "total_input_tokens": 500,
                "total_output_tokens": 200,
                "total_tokens": 700,
                "call_count": 2,
            },
            {
                "model": "claude-sonnet-4.5",
                "provider": "anthropic",
                "total_cost": Decimal("2.00"),
                "total_input_tokens": 1000,
                "total_output_tokens": 400,
                "total_tokens": 1400,
                "call_count": 3,
            },
        ]

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_by_model", new_callable=AsyncMock, return_value=by_model),
            patch("seer.api.usage.router.get_llm_usage_by_operation", new_callable=AsyncMock, return_value=[]),
            patch("seer.api.usage.router.get_llm_usage_daily_trend", new_callable=AsyncMock, return_value=[]),
        ):
            result = await get_usage_analytics(
                request=mock_request,
                start=None,
                end=None,
                model=None,
                operation=None,
            )

        assert result.total_cost == 3.0
        assert result.total_tokens == 2100
        assert result.total_calls == 5


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetWorkflowCosts:
    """Tests for GET /api/usage/analytics/workflows endpoint."""

    async def test_workflow_breakdown(self, mock_request):
        """Test per-workflow cost breakdown."""
        from seer.api.usage.router import get_workflow_costs

        workflows = [
            {
                "workflow_id": "wf_1",
                "workflow_name": "Email Workflow",
                "total_cost": Decimal("2.50"),
                "total_tokens": 5000,
                "call_count": 10,
            },
            {
                "workflow_id": "wf_2",
                "workflow_name": "Data Pipeline",
                "total_cost": Decimal("1.00"),
                "total_tokens": 2000,
                "call_count": 4,
            },
        ]

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_by_workflow", new_callable=AsyncMock, return_value=workflows),
        ):
            result = await get_workflow_costs(request=mock_request, start=None, end=None)

        assert result.period_start == PERIOD_START
        assert result.period_end == PERIOD_END
        assert len(result.workflows) == 2
        assert result.workflows[0].workflow_id == "wf_1"
        assert result.workflows[0].workflow_name == "Email Workflow"
        assert result.workflows[1].total_cost == 1.0

    async def test_empty_workflows(self, mock_request):
        """Test empty workflow costs."""
        from seer.api.usage.router import get_workflow_costs

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_by_workflow", new_callable=AsyncMock, return_value=[]),
        ):
            result = await get_workflow_costs(request=mock_request, start=None, end=None)

        assert result.workflows == []

    async def test_workflow_name_resolution(self, mock_request):
        """Test workflow name is included in response."""
        from seer.api.usage.router import get_workflow_costs

        workflows = [
            {
                "workflow_id": "wf_42",
                "workflow_name": "My Custom Workflow",
                "total_cost": Decimal("5.00"),
                "total_tokens": 10000,
                "call_count": 20,
            },
        ]

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_by_workflow", new_callable=AsyncMock, return_value=workflows),
        ):
            result = await get_workflow_costs(request=mock_request, start=None, end=None)

        assert result.workflows[0].workflow_name == "My Custom Workflow"
        assert result.workflows[0].workflow_id == "wf_42"


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetUsageRecords:
    """Tests for GET /api/usage/analytics/records endpoint."""

    async def test_paginated_records(self, mock_request):
        """Test paginated record response."""
        from seer.api.usage.router import get_usage_records

        records = [
            {
                "id": 1,
                "provider": "openai",
                "model": "gpt-4o",
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
                "cost": Decimal("0.05"),
                "operation": "workflow_execution",
                "workflow_run_id": "run_1",
                "created_at": datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
            },
        ]

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_records_paginated", new_callable=AsyncMock, return_value=(records, 25)),
        ):
            result = await get_usage_records(
                request=mock_request, start=None, end=None, limit=10, offset=0,
            )

        assert result.total == 25
        assert result.limit == 10
        assert result.offset == 0
        assert len(result.records) == 1
        assert result.records[0].model == "gpt-4o"
        assert result.records[0].cost == 0.05
        assert result.records[0].operation == "workflow_execution"

    async def test_default_pagination_params(self, mock_request):
        """Test default limit and offset values."""
        from seer.api.usage.router import get_usage_records

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_records_paginated", new_callable=AsyncMock, return_value=([], 0)) as mock_fn,
        ):
            result = await get_usage_records(
                request=mock_request, start=None, end=None, limit=50, offset=0,
            )

        assert result.limit == 50
        assert result.offset == 0
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["limit"] == 50
        assert call_kwargs["offset"] == 0

    async def test_record_format(self, mock_request):
        """Test individual record fields are correct."""
        from seer.api.usage.router import get_usage_records

        created = datetime(2025, 1, 20, 15, 30, tzinfo=timezone.utc)
        records = [
            {
                "id": 42,
                "provider": "anthropic",
                "model": "claude-sonnet-4.5",
                "input_tokens": 2000,
                "output_tokens": 800,
                "total_tokens": 2800,
                "cost": Decimal("0.10"),
                "operation": "chat_message",
                "workflow_run_id": None,
                "created_at": created,
            },
        ]

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_records_paginated", new_callable=AsyncMock, return_value=(records, 1)),
        ):
            result = await get_usage_records(
                request=mock_request, start=None, end=None, limit=50, offset=0,
            )

        rec = result.records[0]
        assert rec.id == 42
        assert rec.provider == "anthropic"
        assert rec.model == "claude-sonnet-4.5"
        assert rec.input_tokens == 2000
        assert rec.output_tokens == 800
        assert rec.total_tokens == 2800
        assert rec.cost == 0.10
        assert rec.operation == "chat_message"
        assert rec.workflow_run_id is None
        assert rec.created_at == created

    async def test_empty_records(self, mock_request):
        """Test empty paginated response."""
        from seer.api.usage.router import get_usage_records

        with (
            patch("seer.api.usage.router._resolve_period", new_callable=AsyncMock, return_value=(PERIOD_START, PERIOD_END)),
            patch("seer.api.usage.router.get_llm_usage_records_paginated", new_callable=AsyncMock, return_value=([], 0)),
        ):
            result = await get_usage_records(
                request=mock_request, start=None, end=None, limit=50, offset=0,
            )

        assert result.records == []
        assert result.total == 0


@pytest.mark.asyncio
@pytest.mark.unit
class TestResolvePeriod:
    """Tests for _resolve_period helper."""

    async def test_uses_provided_start_and_end(self, mock_user):
        """Test that provided start/end are returned as-is."""
        from seer.api.usage.router import _resolve_period

        start = datetime(2025, 3, 1, tzinfo=timezone.utc)
        end = datetime(2025, 4, 1, tzinfo=timezone.utc)

        result = await _resolve_period(mock_user, start, end)
        assert result == (start, end)

    async def test_defaults_to_billing_period(self, mock_user):
        """Test that None values default to billing period."""
        from seer.api.usage.router import _resolve_period

        with patch(
            "seer.api.usage.router.get_billing_period_for_user",
            new_callable=AsyncMock,
            return_value=(PERIOD_START, PERIOD_END),
        ):
            result = await _resolve_period(mock_user, None, None)

        assert result == (PERIOD_START, PERIOD_END)

    async def test_partial_start_provided(self, mock_user):
        """Test that only start provided uses billing period end."""
        from seer.api.usage.router import _resolve_period

        custom_start = datetime(2025, 1, 15, tzinfo=timezone.utc)

        with patch(
            "seer.api.usage.router.get_billing_period_for_user",
            new_callable=AsyncMock,
            return_value=(PERIOD_START, PERIOD_END),
        ):
            result = await _resolve_period(mock_user, custom_start, None)

        assert result == (custom_start, PERIOD_END)

    async def test_partial_end_provided(self, mock_user):
        """Test that only end provided uses billing period start."""
        from seer.api.usage.router import _resolve_period

        custom_end = datetime(2025, 1, 20, tzinfo=timezone.utc)

        with patch(
            "seer.api.usage.router.get_billing_period_for_user",
            new_callable=AsyncMock,
            return_value=(PERIOD_START, PERIOD_END),
        ):
            result = await _resolve_period(mock_user, None, custom_end)

        assert result == (PERIOD_START, custom_end)
