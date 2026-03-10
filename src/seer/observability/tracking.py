"""
Usage tracking service for monitoring resource consumption.

Provides functions to:
- Increment usage counters
- Query current usage across different time periods
- Track LLM usage and costs
- Support Valkey/Redis caching for performance
- Query analytics breakdowns (by model, operation, daily trend, workflow)
- Handle overage tracking and Stripe reporting
"""
# pylint: disable=too-many-lines  # Tracking module aggregates related usage functions
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from tortoise.expressions import F
from tortoise.functions import Count, Sum

from seer.database.models import User
from seer.database.organization_models import Organization, OrganizationType
from seer.database.overage_models import OverageSettings
from seer.database.usage_models import (
    LLMUsageRecord,
    ResourceType,
    UsageCounter,
)
from seer.database import Workflow, WorkflowRun, WorkflowRunStatus, make_workflow_public_id, parse_run_public_id
from seer.observability.service import (
    get_billing_period_for_org,
    get_billing_period_for_user,
    get_effective_billing_period,
    get_limits_for_org,
    get_limits_for_user,
)
from seer.logger import get_logger

logger = get_logger(__name__)

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
    organization: Optional[Organization] = None,
) -> LLMUsageRecord:
    """
    Track an LLM API call for cost monitoring.

    For team organizations, this updates both user-in-org counters (for per-member
    breakdown) and org-level counters (for limit enforcement).

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
        organization: Optional organization context (for team billing)

    Returns:
        The created LLMUsageRecord
    """
    logger.info(
        "Tracking LLM usage for user %s (org=%s): provider=%s, model=%s, input_tokens=%d, output_tokens=%d, cost=%.6f",
        user.user_id,
        organization.id if organization else None,
        provider,
        model,
        input_tokens,
        output_tokens,
        cost,
    )
    record = await LLMUsageRecord.create(
        user=user,
        organization=organization,  # Set org FK for team-level tracking
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

    # Get billing period based on context
    period_start, period_end = await get_effective_billing_period(user, organization)

    # 1. Update user-in-org counter (tracks individual member's contribution)
    user_counter, _ = await UsageCounter.get_or_create(
        user=user,
        organization=organization,
        resource_type=ResourceType.LLM_CREDITS,
        period_start=period_start,
        period_end=period_end,
        defaults={"count": 0, "value": Decimal("0.0")},
    )

    # Increment cost atomically
    await UsageCounter.filter(id=user_counter.id).update(
        count=F("count") + 1,  # Count of API calls
        value=F("value") + cost,  # Total cost
    )

    # 2. For team orgs, also update org-level aggregate counter (user=None)
    if organization and organization.type == OrganizationType.TEAM:
        org_counter, _ = await UsageCounter.get_or_create(
            user=None,  # Org-level aggregate
            organization=organization,
            resource_type=ResourceType.LLM_CREDITS,
            period_start=period_start,
            period_end=period_end,
            defaults={"count": 0, "value": Decimal("0.0")},
        )
        await UsageCounter.filter(id=org_counter.id).update(
            count=F("count") + 1,
            value=F("value") + cost,
        )

    # Check if this usage triggers overage billing
    await _handle_potential_overage(user, record, cost, organization)

    return record


async def _get_user_overage_settings(user: User) -> Optional[OverageSettings]:
    """
    Get enabled overage settings for a user's personal organization.

    Args:
        user: The user to check.

    Returns:
        OverageSettings if enabled, None otherwise.
    """
    try:
        # Find user's personal organization
        personal_org = await Organization.get_or_none(
            owner=user,
            type=OrganizationType.PERSONAL
        )
        if not personal_org:
            return None

        overage_settings = await OverageSettings.get_or_none(
            organization=personal_org,
            enabled=True,
        )
        return overage_settings
    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation
        return None


async def _get_org_overage_settings(organization: Organization) -> Optional[OverageSettings]:
    """
    Get enabled overage settings for an organization.

    Args:
        organization: The organization to check.

    Returns:
        OverageSettings if enabled, None otherwise.
    """
    try:
        overage_settings = await OverageSettings.get_or_none(
            organization=organization,
            enabled=True,
        )
        return overage_settings
    except Exception:  # pylint: disable=broad-except  # reason: graceful degradation
        return None


async def _get_effective_overage_settings(
    user: User,
    organization: Optional[Organization],
) -> Optional[OverageSettings]:
    """
    Get effective overage settings based on organization context.

    Args:
        user: The user context.
        organization: Optional organization context.

    Returns:
        OverageSettings if enabled, None otherwise.
    """
    if organization and organization.type == OrganizationType.TEAM:
        return await _get_org_overage_settings(organization)
    return await _get_user_overage_settings(user)


async def _handle_potential_overage(
    user: User,
    llm_record: LLMUsageRecord,
    cost: Decimal,
    organization: Optional[Organization] = None,
) -> None:
    """
    Handle potential overage billing after LLM usage is tracked.

    Called after each LLM usage record is created. Determines if the usage
    pushes the user/org into overage territory and reports to Stripe if so.

    For team orgs, uses organization's billing profile and org-level usage.

    Args:
        user: The user who incurred the cost.
        llm_record: The LLM usage record that was created.
        cost: The cost in USD.
        organization: Optional organization context (for team billing).
    """
    # Get effective overage settings (from org for team orgs, user otherwise)
    overage_settings = await _get_effective_overage_settings(user, organization)
    if not overage_settings:
        return

    # Get effective limits (org limits for team orgs, user limits otherwise)
    if organization and organization.type == OrganizationType.TEAM:
        limits = await get_limits_for_org(organization)
    else:
        limits = await get_limits_for_user(user)

    if limits.has_unlimited_credits:
        return

    monthly_limit = Decimal(str(limits.llm_credits_monthly))

    # Get current monthly usage (org-level for team orgs, user-level otherwise)
    if organization and organization.type == OrganizationType.TEAM:
        current_usage = await get_org_monthly_llm_credits_used(organization)
    else:
        current_usage = await get_monthly_llm_credits_used(user)

    # Check if we're in overage territory (over 100% of subscription limit)
    if current_usage <= monthly_limit:
        return

    # Calculate overage portion of this usage
    # The overage is the amount over the subscription limit
    previous_usage = current_usage - cost
    overage_start = max(monthly_limit, previous_usage)
    overage_amount = current_usage - overage_start

    if overage_amount <= 0:
        return

    # Calculate billed amount with margin
    margin = overage_settings.margin_multiplier
    base_cost_cents = int(overage_amount * 100)
    billed_amount_cents = int(float(overage_amount) * float(margin) * 100)

    # Report to Stripe
    await _report_overage_to_stripe(overage_settings, llm_record, base_cost_cents, billed_amount_cents)


async def _report_overage_to_stripe(
    overage_settings: OverageSettings,
    llm_record: LLMUsageRecord,
    base_cost_cents: int,
    billed_amount_cents: int,
) -> None:
    """
    Report overage usage to Stripe.

    Args:
        overage_settings: The overage settings for the user.
        llm_record: The LLM usage record that triggered the overage.
        base_cost_cents: The actual LLM cost in cents.
        billed_amount_cents: The billed amount (cost × margin) in cents.
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular import
    from seer.api.subscriptions.overage_service import report_usage_to_stripe

    await report_usage_to_stripe(
        overage_settings=overage_settings,
        llm_record=llm_record,
        base_cost_cents=base_cost_cents,
        billed_amount_cents=billed_amount_cents,
    )


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


async def get_rolling_llm_credits_used(user: User, window: timedelta) -> Decimal:
    """
    Get total LLM credits used in a rolling time window.

    Queries LLMUsageRecord directly using the (user_id, created_at) index
    for efficient rolling window calculations.

    Args:
        user: The user to get credits for
        window: The time window to query (e.g., timedelta(hours=5))

    Returns:
        Total credits used in USD during the rolling window
    """
    cutoff = datetime.now(timezone.utc) - window
    result = await LLMUsageRecord.filter(
        user=user,
        created_at__gte=cutoff,
    ).annotate(total_cost=Sum("cost")).values("total_cost")

    if result and result[0]["total_cost"]:
        return Decimal(str(result[0]["total_cost"]))
    return Decimal("0.0")


async def get_5h_llm_credits_used(user: User) -> Decimal:
    """
    Get total LLM credits used in the last 5 hours.

    Uses a rolling window for burst protection.

    Args:
        user: The user to get credits for

    Returns:
        Total credits used in USD during the last 5 hours
    """
    return await get_rolling_llm_credits_used(user, timedelta(hours=5))


async def get_weekly_llm_credits_used(user: User) -> Decimal:
    """
    Get total LLM credits used in the last 7 days.

    Uses a rolling window to prevent front-loading usage.

    Args:
        user: The user to get credits for

    Returns:
        Total credits used in USD during the last 7 days
    """
    return await get_rolling_llm_credits_used(user, timedelta(days=7))


# =============================================================================
# Organization-Scoped Query Functions
# =============================================================================


async def get_org_workflow_count(organization: Organization) -> int:
    """
    Get the total workflow count for an organization.

    Args:
        organization: The organization to get count for

    Returns:
        Total workflow count
    """
    count = await Workflow.filter(organization=organization).count()
    return count


async def get_org_monthly_run_count(organization: Organization) -> int:
    """
    Get the workflow run count for the current month for an organization.

    Args:
        organization: The organization to get count for

    Returns:
        Monthly run count
    """
    period_start, period_end = await get_billing_period_for_org(organization)

    count = await WorkflowRun.filter(
        workflow__organization=organization,
        created_at__gte=period_start,
        created_at__lt=period_end,
        status=WorkflowRunStatus.SUCCEEDED,
    ).count()

    return count


async def get_org_monthly_llm_credits_used(organization: Organization) -> Decimal:
    """
    Get the total LLM credits used this month for an organization.

    Queries the org-level UsageCounter (user=None, organization=org).

    Args:
        organization: The organization to get credits for

    Returns:
        Total credits used in USD
    """
    period_start, period_end = await get_billing_period_for_org(organization)

    counter = await UsageCounter.get_or_none(
        user=None,  # Org-level aggregate
        organization=organization,
        resource_type=ResourceType.LLM_CREDITS,
        period_start=period_start,
        period_end=period_end,
    )

    return counter.value if counter else Decimal("0.0")


async def get_org_rolling_llm_credits_used(organization: Organization, window: timedelta) -> Decimal:
    """
    Get total LLM credits used in a rolling time window for an organization.

    Queries LLMUsageRecord directly using the (organization_id, created_at) index
    for efficient rolling window calculations.

    Args:
        organization: The organization to get credits for
        window: The time window to query (e.g., timedelta(hours=5))

    Returns:
        Total credits used in USD during the rolling window
    """
    cutoff = datetime.now(timezone.utc) - window
    result = await LLMUsageRecord.filter(
        organization=organization,
        created_at__gte=cutoff,
    ).annotate(total_cost=Sum("cost")).values("total_cost")

    if result and result[0]["total_cost"]:
        return Decimal(str(result[0]["total_cost"]))
    return Decimal("0.0")


async def get_org_5h_llm_credits_used(organization: Organization) -> Decimal:
    """
    Get total LLM credits used in the last 5 hours for an organization.

    Uses a rolling window for burst protection.

    Args:
        organization: The organization to get credits for

    Returns:
        Total credits used in USD during the last 5 hours
    """
    return await get_org_rolling_llm_credits_used(organization, timedelta(hours=5))


async def get_org_weekly_llm_credits_used(organization: Organization) -> Decimal:
    """
    Get total LLM credits used in the last 7 days for an organization.

    Uses a rolling window to prevent front-loading usage.

    Args:
        organization: The organization to get credits for

    Returns:
        Total credits used in USD during the last 7 days
    """
    return await get_org_rolling_llm_credits_used(organization, timedelta(days=7))


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
