"""
Unit tests for analytics query functions in seer.observability.tracking.
"""
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.observability.tracking import (
    get_llm_usage_by_model,
    get_llm_usage_by_operation,
    get_llm_usage_by_workflow,
    get_llm_usage_daily_trend,
    get_llm_usage_records_paginated,
)


PERIOD_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
PERIOD_END = datetime(2025, 2, 1, tzinfo=timezone.utc)


def _make_annotate_chain(return_values):
    """Build a mock queryset that supports .group_by().annotate().values() chaining."""
    mock_qs = MagicMock()
    mock_qs.filter.return_value = mock_qs
    mock_qs.group_by.return_value = mock_qs
    mock_qs.annotate.return_value = mock_qs
    mock_qs.values = AsyncMock(return_value=return_values)
    return mock_qs


def _make_values_chain(return_values):
    """Build a mock queryset for .filter().values() (no group_by)."""
    mock_qs = MagicMock()
    mock_qs.filter.return_value = mock_qs
    mock_qs.values = AsyncMock(return_value=return_values)
    return mock_qs


def _make_paginated_chain(return_values, count_value):
    """Build a mock queryset for paginated records with .count(), .order_by().offset().limit().values()."""
    mock_qs = MagicMock()
    mock_qs.filter.return_value = mock_qs
    mock_qs.count = AsyncMock(return_value=count_value)
    mock_qs.order_by.return_value = mock_qs
    mock_qs.offset.return_value = mock_qs
    mock_qs.limit.return_value = mock_qs
    mock_qs.values = AsyncMock(return_value=return_values)
    return mock_qs


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetLlmUsageByModel:
    """Tests for get_llm_usage_by_model."""

    async def test_groups_by_model_and_provider(self, mock_user):
        """Test that results are grouped by model and provider."""
        mock_data = [
            {
                "model": "gpt-4o",
                "provider": "openai",
                "total_cost": Decimal("1.50"),
                "total_input_tokens": 1000,
                "total_output_tokens": 500,
                "total_tokens": 1500,
                "call_count": 3,
            },
            {
                "model": "claude-sonnet-4.5",
                "provider": "anthropic",
                "total_cost": Decimal("2.00"),
                "total_input_tokens": 2000,
                "total_output_tokens": 800,
                "total_tokens": 2800,
                "call_count": 5,
            },
        ]
        mock_qs = _make_annotate_chain(mock_data)

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_by_model(mock_user, PERIOD_START, PERIOD_END)

        assert len(result) == 2
        assert result[0]["model"] == "gpt-4o"
        assert result[1]["model"] == "claude-sonnet-4.5"
        assert result[0]["call_count"] == 3
        assert result[1]["total_cost"] == Decimal("2.00")

    async def test_empty_results(self, mock_user):
        """Test empty period returns empty list."""
        mock_qs = _make_annotate_chain([])

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_by_model(mock_user, PERIOD_START, PERIOD_END)

        assert result == []

    async def test_filter_by_model(self, mock_user):
        """Test filtering by specific model."""
        mock_data = [
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
        mock_qs = _make_annotate_chain(mock_data)
        mock_qs.filter.return_value = mock_qs  # Chained filter

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_by_model(mock_user, PERIOD_START, PERIOD_END, model="gpt-4o")

        assert len(result) == 1
        assert result[0]["model"] == "gpt-4o"

    async def test_filter_by_operation(self, mock_user):
        """Test filtering by operation type."""
        mock_data = [
            {
                "model": "gpt-4o",
                "provider": "openai",
                "total_cost": Decimal("0.50"),
                "total_input_tokens": 500,
                "total_output_tokens": 200,
                "total_tokens": 700,
                "call_count": 1,
            },
        ]
        mock_qs = _make_annotate_chain(mock_data)
        mock_qs.filter.return_value = mock_qs

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_by_model(mock_user, PERIOD_START, PERIOD_END, operation="chat_message")

        assert len(result) == 1


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetLlmUsageByOperation:
    """Tests for get_llm_usage_by_operation."""

    async def test_groups_by_operation(self, mock_user):
        """Test grouping by operation type."""
        mock_data = [
            {
                "operation": "workflow_execution",
                "total_cost": Decimal("3.00"),
                "total_input_tokens": 5000,
                "total_output_tokens": 2000,
                "total_tokens": 7000,
                "call_count": 10,
            },
            {
                "operation": "chat_message",
                "total_cost": Decimal("1.00"),
                "total_input_tokens": 2000,
                "total_output_tokens": 800,
                "total_tokens": 2800,
                "call_count": 4,
            },
            {
                "operation": "browser_execution",
                "total_cost": Decimal("0.50"),
                "total_input_tokens": 1000,
                "total_output_tokens": 300,
                "total_tokens": 1300,
                "call_count": 2,
            },
        ]
        mock_qs = _make_annotate_chain(mock_data)

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_by_operation(mock_user, PERIOD_START, PERIOD_END)

        assert len(result) == 3
        ops = {r["operation"] for r in result}
        assert ops == {"workflow_execution", "chat_message", "browser_execution"}

    async def test_empty_results(self, mock_user):
        """Test empty period returns empty list."""
        mock_qs = _make_annotate_chain([])

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_by_operation(mock_user, PERIOD_START, PERIOD_END)

        assert result == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetLlmUsageDailyTrend:
    """Tests for get_llm_usage_daily_trend."""

    async def test_daily_aggregation(self, mock_user):
        """Test records are aggregated by day."""
        mock_data = [
            {"created_at": datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc), "cost": Decimal("0.50"), "total_tokens": 500},
            {"created_at": datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc), "cost": Decimal("0.30"), "total_tokens": 300},
            {"created_at": datetime(2025, 1, 2, 9, 0, tzinfo=timezone.utc), "cost": Decimal("1.00"), "total_tokens": 1000},
        ]
        mock_qs = _make_values_chain(mock_data)
        mock_qs.filter.return_value = mock_qs

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_daily_trend(mock_user, PERIOD_START, PERIOD_END)

        assert len(result) == 2
        # Should be sorted ascending
        assert result[0]["date"] == "2025-01-01"
        assert result[1]["date"] == "2025-01-02"
        # Day 1: two records aggregated
        assert result[0]["total_cost"] == Decimal("0.80")
        assert result[0]["total_tokens"] == 800
        assert result[0]["call_count"] == 2
        # Day 2: single record
        assert result[1]["total_cost"] == Decimal("1.00")
        assert result[1]["call_count"] == 1

    async def test_sorted_ascending(self, mock_user):
        """Test results sorted by date ascending even when input is unordered."""
        mock_data = [
            {"created_at": datetime(2025, 1, 5, 10, 0, tzinfo=timezone.utc), "cost": Decimal("0.10"), "total_tokens": 100},
            {"created_at": datetime(2025, 1, 3, 10, 0, tzinfo=timezone.utc), "cost": Decimal("0.20"), "total_tokens": 200},
            {"created_at": datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc), "cost": Decimal("0.30"), "total_tokens": 300},
        ]
        mock_qs = _make_values_chain(mock_data)
        mock_qs.filter.return_value = mock_qs

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_daily_trend(mock_user, PERIOD_START, PERIOD_END)

        dates = [r["date"] for r in result]
        assert dates == ["2025-01-01", "2025-01-03", "2025-01-05"]

    async def test_filter_by_model(self, mock_user):
        """Test filtering by model narrows results."""
        mock_data = [
            {"created_at": datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc), "cost": Decimal("0.50"), "total_tokens": 500},
        ]
        mock_qs = _make_values_chain(mock_data)
        mock_qs.filter.return_value = mock_qs

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_daily_trend(mock_user, PERIOD_START, PERIOD_END, model="gpt-4o")

        assert len(result) == 1

    async def test_empty_results(self, mock_user):
        """Test empty period returns empty list."""
        mock_qs = _make_values_chain([])
        mock_qs.filter.return_value = mock_qs

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_daily_trend(mock_user, PERIOD_START, PERIOD_END)

        assert result == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetLlmUsageByWorkflow:
    """Tests for get_llm_usage_by_workflow."""

    async def test_workflow_resolution_and_aggregation(self, mock_user):
        """Test that run IDs are resolved to workflows and re-aggregated."""
        per_run_data = [
            {"workflow_run_id": "run_10", "total_cost": Decimal("1.00"), "total_tokens": 1000, "call_count": 2},
            {"workflow_run_id": "run_11", "total_cost": Decimal("0.50"), "total_tokens": 500, "call_count": 1},
            # Both run_10 and run_11 belong to same workflow (id=1)
        ]
        mock_qs = _make_annotate_chain(per_run_data)

        # Mock WorkflowRun query
        # Note: MagicMock(name=...) sets the mock's repr name, not a .name attribute.
        # We must set .name explicitly after construction.
        mock_wf = MagicMock(id=1)
        mock_wf.name = "My Workflow"

        mock_run_10 = MagicMock()
        mock_run_10.id = 10
        mock_run_10.workflow = mock_wf

        mock_run_11 = MagicMock()
        mock_run_11.id = 11
        mock_run_11.workflow = mock_wf

        mock_run_qs = MagicMock()
        mock_run_qs.prefetch_related = AsyncMock(return_value=[mock_run_10, mock_run_11])

        with (
            patch("seer.observability.tracking.LLMUsageRecord") as mock_model,
            patch("seer.observability.tracking.WorkflowRun") as mock_wf_run,
        ):
            mock_model.filter.return_value = mock_qs
            mock_wf_run.filter.return_value = mock_run_qs
            result = await get_llm_usage_by_workflow(mock_user, PERIOD_START, PERIOD_END)

        assert len(result) == 1
        assert result[0]["workflow_id"] == "wf_1"
        assert result[0]["workflow_name"] == "My Workflow"
        # Re-aggregated: 1.00 + 0.50
        assert result[0]["total_cost"] == Decimal("1.50")
        assert result[0]["total_tokens"] == 1500
        assert result[0]["call_count"] == 3

    async def test_multiple_workflows(self, mock_user):
        """Test aggregation across multiple workflows."""
        per_run_data = [
            {"workflow_run_id": "run_10", "total_cost": Decimal("1.00"), "total_tokens": 1000, "call_count": 2},
            {"workflow_run_id": "run_20", "total_cost": Decimal("2.00"), "total_tokens": 2000, "call_count": 4},
        ]
        mock_qs = _make_annotate_chain(per_run_data)

        mock_wf_a = MagicMock(id=1)
        mock_wf_a.name = "Workflow A"
        mock_wf_b = MagicMock(id=2)
        mock_wf_b.name = "Workflow B"

        mock_run_10 = MagicMock()
        mock_run_10.id = 10
        mock_run_10.workflow = mock_wf_a

        mock_run_20 = MagicMock()
        mock_run_20.id = 20
        mock_run_20.workflow = mock_wf_b

        mock_run_qs = MagicMock()
        mock_run_qs.prefetch_related = AsyncMock(return_value=[mock_run_10, mock_run_20])

        with (
            patch("seer.observability.tracking.LLMUsageRecord") as mock_model,
            patch("seer.observability.tracking.WorkflowRun") as mock_wf_run,
        ):
            mock_model.filter.return_value = mock_qs
            mock_wf_run.filter.return_value = mock_run_qs
            result = await get_llm_usage_by_workflow(mock_user, PERIOD_START, PERIOD_END)

        assert len(result) == 2
        wf_ids = {r["workflow_id"] for r in result}
        assert wf_ids == {"wf_1", "wf_2"}

    async def test_empty_results(self, mock_user):
        """Test empty period returns empty list."""
        mock_qs = _make_annotate_chain([])

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_by_workflow(mock_user, PERIOD_START, PERIOD_END)

        assert result == []

    async def test_invalid_run_ids_skipped(self, mock_user):
        """Test that invalid run IDs are gracefully skipped."""
        per_run_data = [
            {"workflow_run_id": "invalid_format", "total_cost": Decimal("1.00"), "total_tokens": 1000, "call_count": 2},
        ]
        mock_qs = _make_annotate_chain(per_run_data)

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            result = await get_llm_usage_by_workflow(mock_user, PERIOD_START, PERIOD_END)

        # Invalid run ID means no runs to resolve → empty result
        assert result == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestGetLlmUsageRecordsPaginated:
    """Tests for get_llm_usage_records_paginated."""

    async def test_pagination(self, mock_user):
        """Test paginated results with total count."""
        mock_data = [
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
        mock_qs = _make_paginated_chain(mock_data, count_value=25)

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            records, total = await get_llm_usage_records_paginated(
                mock_user, PERIOD_START, PERIOD_END, limit=10, offset=0,
            )

        assert total == 25
        assert len(records) == 1
        assert records[0]["model"] == "gpt-4o"

    async def test_ordering(self, mock_user):
        """Test records are ordered by -created_at."""
        mock_qs = _make_paginated_chain([], count_value=0)

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            await get_llm_usage_records_paginated(mock_user, PERIOD_START, PERIOD_END)

        mock_qs.order_by.assert_called_once_with("-created_at")

    async def test_default_limit_and_offset(self, mock_user):
        """Test default pagination params."""
        mock_qs = _make_paginated_chain([], count_value=0)

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            records, total = await get_llm_usage_records_paginated(mock_user, PERIOD_START, PERIOD_END)

        assert total == 0
        assert records == []
        mock_qs.offset.assert_called_once_with(0)
        mock_qs.limit.assert_called_once_with(50)

    async def test_empty_results(self, mock_user):
        """Test empty period returns empty list and zero count."""
        mock_qs = _make_paginated_chain([], count_value=0)

        with patch("seer.observability.tracking.LLMUsageRecord") as mock_model:
            mock_model.filter.return_value = mock_qs
            records, total = await get_llm_usage_records_paginated(mock_user, PERIOD_START, PERIOD_END)

        assert records == []
        assert total == 0
