"""
Usage tracking service for monitoring resource consumption.

Provides functions to:
- Increment usage counters
- Query current usage across different time periods
- Track LLM usage and costs
- Support Redis caching for performance
"""
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from tortoise.expressions import F
from tortoise.functions import Sum

from seer.database.models import User
from seer.database.usage_models import (
    LLMUsageRecord,
    ResourceType,
    UsageCounter,
)
from seer.database import Workflow, WorkflowRun, WorkflowRunStatus
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
async def get_chat_message_count(user: User, workflow_id: str) -> int:
    """
    DEPRECATED: Use get_total_chat_message_count() instead.

    This function is kept for backwards compatibility but will be removed.
    Chat limits are now tracked globally per user, not per workflow.
    """
    # For backwards compat, just return global count
    return await get_total_chat_message_count(user)


async def track_llm_usage(
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
