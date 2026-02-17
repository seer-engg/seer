"""
Usage tracking service for monitoring resource consumption.

Provides functions to:
- Increment usage counters
- Query current usage across different time periods
- Track LLM usage and costs
- Support Valkey/Redis caching for performance
- Query analytics breakdowns (by model, operation, daily trend, workflow)
"""
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from tortoise.expressions import F
from tortoise.functions import Count, Sum

from seer.database.models import User
from seer.database.usage_models import (
    LLMUsageRecord,
    ResourceType,
    UsageCounter,
)
from seer.database import Workflow, WorkflowRun, WorkflowRunStatus, make_workflow_public_id, parse_run_public_id
from seer.observability.service import get_billing_period_for_user
from seer.logger import get_logger

logger = get_logger(__name__)

async def increment_chat_message_count(user: User) -> int:
    """
    Increment the global chat message count for a user.

    Changed from per-workflow to global tracking (across all workflows).

    Args:
        user: The user to increment count for

    Returns:
        The new total message count for this user
    """
    counter, _ = await UsageCounter.get_or_create(
        user=user,
        resource_type=ResourceType.CHAT_MESSAGES,
        reference_id=None,  # Global, not per-workflow
        period_start=None,  # All-time counter
        period_end=None,
        defaults={"count": 0},
    )

    await UsageCounter.filter(id=counter.id).update(count=F("count") + 1)

    await counter.refresh_from_db()
    return counter.count


async def get_workflow_count(user: User) -> int:
    """
    Get the total workflow count for a user.

    Args:
        user: The user to get count for

    Returns:
        Total workflow count
    """
    count = await Workflow.filter(user=user).count()
    return count


async def get_monthly_run_count(user: User) -> int:
    """
    Get the workflow run count for the current month.

    Args:
        user: The user to get count for

    Returns:
        Monthly run count
    """
    period_start, period_end = await get_billing_period_for_user(user)

    count = await WorkflowRun.filter(
        user=user,
        created_at__gte=period_start,
        created_at__lt=period_end,
        status= WorkflowRunStatus.SUCCEEDED
    ).count()

    return count


async def get_total_chat_message_count(user: User) -> int:
    """
    Get the total chat message count for a user across ALL workflows.

    Changed from per-workflow to global tracking.

    Args:
        user: The user to get count for

    Returns:
        Total message count across all workflows
    """
    counter = await UsageCounter.get_or_none(
        user=user,
        resource_type=ResourceType.CHAT_MESSAGES,
        reference_id=None,  # Global, not per-workflow
        period_start=None,
        period_end=None,
    )

    return counter.count if counter else 0


# Deprecated: Use get_total_chat_message_count instead
async def get_chat_message_count(user: User) -> int:
    """
    DEPRECATED: Use get_total_chat_message_count() instead.

    This function is kept for backwards compatibility but will be removed.
    Chat limits are now tracked globally per user, not per workflow.
    """
    # For backwards compat, just return global count
    return await get_total_chat_message_count(user)


async def track_llm_usage(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # Reason: essential parameters for LLM tracking
    user: User,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: Decimal,
    workflow_run_id: Optional[str] = None,
    operation: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> LLMUsageRecord:
    """
    Track an LLM API call for cost monitoring.

    Args:
        user: The user making the call
        provider: LLM provider (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3-opus")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost: Cost in USD
        workflow_run_id: Optional workflow run ID
        operation: Optional operation type (e.g., "workflow_execution", "chat_message")
        metadata: Optional additional metadata

    Returns:
        The created LLMUsageRecord
    """
    logger.info(
        "Tracking LLM usage for user %s: provider=%s, model=%s, input_tokens=%d, output_tokens=%d, cost=%.6f",
        user.user_id,
        provider,
        model,
        input_tokens,
        output_tokens,
        cost,
    )
    record = await LLMUsageRecord.create(
        user=user,
        workflow_run_id=workflow_run_id,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost=cost,
        operation=operation,
        metadata=metadata,
    )

    # Also update monthly credit counter aligned to billing period
    period_start, period_end = await get_billing_period_for_user(user)

    counter, _ = await UsageCounter.get_or_create(
        user=user,
        resource_type=ResourceType.LLM_CREDITS,
        period_start=period_start,
        period_end=period_end,
        defaults={"count": 0, "value": Decimal("0.0")},
    )

    # Increment cost atomically
    await UsageCounter.filter(id=counter.id).update(
        count=F("count") + 1,  # Count of API calls
        value=F("value") + cost,  # Total cost
    )

    return record


async def get_monthly_llm_credits_used(user: User) -> Decimal:
    """
    Get the total LLM credits used this month.

    Args:
        user: The user to get credits for

    Returns:
        Total credits used in USD
    """
    period_start, period_end = await get_billing_period_for_user(user)

    counter = await UsageCounter.get_or_none(
        user=user,
        resource_type=ResourceType.LLM_CREDITS,
        period_start=period_start,
        period_end=period_end,
    )

    return counter.value if counter else Decimal("0.0")


async def get_monthly_llm_credits_detailed(user: User) -> dict:
    """
    Get detailed monthly LLM usage breakdown by model.

    Args:
        user: The user to get breakdown for

    Returns:
        Dictionary with breakdown by model
    """
    period_start, period_end = await get_billing_period_for_user(user)

    # Aggregate by model for current month
    records = (
        await LLMUsageRecord.filter(
            user=user,
            created_at__gte=period_start,
            created_at__lt=period_end,
        )
        .group_by("model")
        .annotate(
            total_cost=Sum("cost"),
            total_tokens=Sum("total_tokens"),
            call_count=Sum("id"),  # Count of records
        )
        .values("model", "total_cost", "total_tokens", "call_count")
    )

    return {
        "period_start": period_start,
        "total_cost": await get_monthly_llm_credits_used(user),
        "by_model": records,
    }


async def reset_monthly_counters(user: User, target_month: Optional[datetime] = None) -> None:
    """
    Reset monthly usage counters for a new billing period.

    This is typically called by a background job at the start of each month.
    It doesn't delete old counters (for historical data), just ensures new ones exist.

    Args:
        user: The user to reset counters for
        target_month: The month to reset for (defaults to current month)
    """
    reference_now = target_month
    if reference_now and reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)
    period_start, period_end = await get_billing_period_for_user(
        user, reference_now=reference_now
    )

    # Create new monthly counters with zero values (if they don't exist)
    for resource_type in [ResourceType.RUNS, ResourceType.LLM_CREDITS]:
        await UsageCounter.get_or_create(
            user=user,
            resource_type=resource_type,
            period_start=period_start,
            period_end=period_end,
            defaults={"count": 0, "value": Decimal("0.0")},
        )


# =============================================================================
# Analytics Query Functions
# =============================================================================


async def get_llm_usage_by_model(
    user: User,
    period_start: datetime,
    period_end: datetime,
    *,
    model: Optional[str] = None,
    operation: Optional[str] = None,
    workflow_run_id: Optional[str] = None,
) -> list[dict]:
    """
    Aggregate LLM usage by model and provider.

    Returns:
        List of dicts with model, provider, total_cost, total_input_tokens,
        total_output_tokens, total_tokens, call_count.
    """
    qs = LLMUsageRecord.filter(user=user, created_at__gte=period_start, created_at__lt=period_end)
    if model:
        qs = qs.filter(model=model)
    if operation:
        qs = qs.filter(operation=operation)
    if workflow_run_id:
        qs = qs.filter(workflow_run_id=workflow_run_id)

    records = (
        await qs.group_by("model", "provider")
        .annotate(
            total_cost=Sum("cost"),
            total_input_tokens=Sum("input_tokens"),
            total_output_tokens=Sum("output_tokens"),
            total_tokens=Sum("total_tokens"),
            call_count=Count("id"),
        )
        .values("model", "provider", "total_cost", "total_input_tokens", "total_output_tokens", "total_tokens", "call_count")
    )
    return records


async def get_llm_usage_by_operation(
    user: User,
    period_start: datetime,
    period_end: datetime,
    *,
    model: Optional[str] = None,
    operation: Optional[str] = None,
    workflow_run_id: Optional[str] = None,
) -> list[dict]:
    """
    Aggregate LLM usage by operation type.

    Returns:
        List of dicts with operation, total_cost, total_input_tokens,
        total_output_tokens, total_tokens, call_count.
    """
    qs = LLMUsageRecord.filter(user=user, created_at__gte=period_start, created_at__lt=period_end)
    if model:
        qs = qs.filter(model=model)
    if operation:
        qs = qs.filter(operation=operation)
    if workflow_run_id:
        qs = qs.filter(workflow_run_id=workflow_run_id)

    records = (
        await qs.group_by("operation")
        .annotate(
            total_cost=Sum("cost"),
            total_input_tokens=Sum("input_tokens"),
            total_output_tokens=Sum("output_tokens"),
            total_tokens=Sum("total_tokens"),
            call_count=Count("id"),
        )
        .values("operation", "total_cost", "total_input_tokens", "total_output_tokens", "total_tokens", "call_count")
    )
    return records


async def get_llm_usage_daily_trend(
    user: User,
    period_start: datetime,
    period_end: datetime,
    *,
    model: Optional[str] = None,
    operation: Optional[str] = None,
    workflow_run_id: Optional[str] = None,
) -> list[dict]:
    """
    Get daily cost/token trend aggregated in Python (Tortoise lacks TruncDate).

    Returns:
        List of dicts with date (YYYY-MM-DD), total_cost, total_tokens, call_count,
        sorted by date ascending.
    """
    qs = LLMUsageRecord.filter(user=user, created_at__gte=period_start, created_at__lt=period_end)
    if model:
        qs = qs.filter(model=model)
    if operation:
        qs = qs.filter(operation=operation)
    if workflow_run_id:
        qs = qs.filter(workflow_run_id=workflow_run_id)

    records = await qs.values("created_at", "cost", "total_tokens")

    daily: dict[str, dict] = {}
    for rec in records:
        day = rec["created_at"].strftime("%Y-%m-%d")
        if day not in daily:
            daily[day] = {"date": day, "total_cost": Decimal("0"), "total_tokens": 0, "call_count": 0}
        daily[day]["total_cost"] += rec["cost"]
        daily[day]["total_tokens"] += rec["total_tokens"]
        daily[day]["call_count"] += 1

    return sorted(daily.values(), key=lambda d: d["date"])


def _parse_run_ids(per_run: list[dict]) -> dict[str, int]:
    """Parse public run IDs from aggregated records into WorkflowRun PKs."""
    run_id_to_pk: dict[str, int] = {}
    for entry in per_run:
        run_pub_id = entry["workflow_run_id"]
        try:
            run_id_to_pk[run_pub_id] = parse_run_public_id(run_pub_id)
        except (ValueError, TypeError):
            logger.warning("Could not parse run_id: %s", run_pub_id)
    return run_id_to_pk


async def _resolve_run_to_workflow(run_id_to_pk: dict[str, int]) -> dict[int, tuple[int, str | None]]:
    """Batch-query WorkflowRun rows and map each PK to (workflow_id, workflow_name)."""
    runs = await WorkflowRun.filter(id__in=list(run_id_to_pk.values())).prefetch_related("workflow")
    pk_to_workflow: dict[int, tuple[int, str | None]] = {}
    for run in runs:
        if run.workflow:
            pk_to_workflow[run.id] = (run.workflow.id, run.workflow.name)
    return pk_to_workflow


async def get_llm_usage_by_workflow(
    user: User,
    period_start: datetime,
    period_end: datetime,
) -> list[dict]:
    """
    Aggregate LLM usage by workflow (resolved from workflow_run_id).

    Strategy:
    1. SQL aggregate by workflow_run_id to get per-run costs
    2. Parse run IDs via parse_run_public_id() to get WorkflowRun PKs
    3. Batch query WorkflowRun for (id, workflow_id, workflow.name)
    4. Re-aggregate by workflow_id in Python, include workflow name
    5. Return with public workflow_id (like "wf_42") via make_workflow_public_id()

    Returns:
        List of dicts with workflow_id, workflow_name, total_cost, total_tokens, call_count.
    """
    # Step 1: Aggregate by workflow_run_id
    per_run = (
        await LLMUsageRecord.filter(
            user=user,
            created_at__gte=period_start,
            created_at__lt=period_end,
            workflow_run_id__not_isnull=True,
        )
        .group_by("workflow_run_id")
        .annotate(
            total_cost=Sum("cost"),
            total_tokens=Sum("total_tokens"),
            call_count=Count("id"),
        )
        .values("workflow_run_id", "total_cost", "total_tokens", "call_count")
    )

    if not per_run:
        return []

    # Step 2: Parse run IDs to PKs
    run_id_to_pk = _parse_run_ids(per_run)
    if not run_id_to_pk:
        return []

    # Step 3: Batch query WorkflowRun -> Workflow
    pk_to_workflow = await _resolve_run_to_workflow(run_id_to_pk)

    # Step 4: Re-aggregate by workflow_id
    workflow_agg: dict[int, dict] = defaultdict(lambda: {"total_cost": Decimal("0"), "total_tokens": 0, "call_count": 0, "workflow_name": None})
    for entry in per_run:
        pk = run_id_to_pk.get(entry["workflow_run_id"])
        if pk is None:
            continue
        wf_info = pk_to_workflow.get(pk)
        if wf_info is None:
            continue
        wf_id, wf_name = wf_info
        workflow_agg[wf_id]["total_cost"] += entry["total_cost"]
        workflow_agg[wf_id]["total_tokens"] += entry["total_tokens"]
        workflow_agg[wf_id]["call_count"] += entry["call_count"]
        workflow_agg[wf_id]["workflow_name"] = wf_name

    # Step 5: Return with public IDs
    return [
        {
            "workflow_id": make_workflow_public_id(wf_id),
            "workflow_name": data["workflow_name"],
            "total_cost": data["total_cost"],
            "total_tokens": data["total_tokens"],
            "call_count": data["call_count"],
        }
        for wf_id, data in workflow_agg.items()
    ]


async def get_llm_usage_records_paginated(
    user: User,
    period_start: datetime,
    period_end: datetime,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """
    Get paginated individual LLM usage records for audit/drill-down.

    Returns:
        Tuple of (records list, total_count).
    """
    qs = LLMUsageRecord.filter(user=user, created_at__gte=period_start, created_at__lt=period_end)

    total_count = await qs.count()

    records = (
        await qs.order_by("-created_at")
        .offset(offset)
        .limit(limit)
        .values(
            "id", "provider", "model", "input_tokens", "output_tokens",
            "total_tokens", "cost", "operation", "workflow_run_id", "created_at",
        )
    )

    return records, total_count
