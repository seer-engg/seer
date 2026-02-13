"""
Usage summary endpoints for exposing current limits and consumption.

Provides a consolidated view of usage across workflows, runs, chat messages,
and LLM credits so the frontend can display upgrade nudges and remaining quota.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from seer.config import config
from seer.database import User
from seer.database.subscription_models import SubscriptionStatus, SubscriptionTier
from seer.observability import (
    get_limits_for_user,
    get_llm_usage_by_model,
    get_llm_usage_by_operation,
    get_llm_usage_by_workflow,
    get_llm_usage_daily_trend,
    get_llm_usage_records_paginated,
    get_monthly_llm_credits_used,
    get_monthly_run_count,
    get_subscription_for_user,
    get_total_chat_message_count,
    get_workflow_count,
    resolve_user_tier,
)
from seer.observability.service import get_billing_period_for_user
from seer.observability.models import TierLimits

router = APIRouter(prefix="/usage", tags=["usage"])


class UsageMetric(BaseModel):
    """Represents usage for a single metered dimension."""

    used: float
    limit: Optional[float]
    remaining: Optional[float]
    is_unlimited: bool
    reset_at: Optional[datetime] = None
    disabled: bool = False
    unit: Optional[str] = None


class UsageBreakdown(BaseModel):
    """All usage metrics returned to the client."""

    chat_messages: UsageMetric
    workflow_runs: UsageMetric
    llm_credits: UsageMetric
    workflows: UsageMetric


class SubscriptionSummary(BaseModel):
    """Subscription context included with usage."""

    tier: str
    status: str
    current_period_end: Optional[datetime] = None
    cancel_at_period_end: bool = False


class UsageResponse(BaseModel):
    """Response model for usage summary."""

    tier: str
    is_self_hosted: bool
    subscription: SubscriptionSummary
    limits: dict[str, Union[int, float]]
    usage: UsageBreakdown

    class Config:
        json_encoders = {Decimal: float}


def _require_user(request: Request) -> User:
    """Extract authenticated user from request or raise 401."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _build_usage_metric(
    *,
    used: Union[int, float, Decimal],
    limit_value: Union[int, float],
    is_unlimited: bool,
    reset_at: Optional[datetime] = None,
    disabled: bool = False,
    unit: Optional[str] = None,
) -> UsageMetric:
    """
    Normalize usage values into a UsageMetric object.

    Handles unlimited limits (-1), disabled features (limit=0), and remaining math.
    """
    used_value = float(used) if isinstance(used, Decimal) else float(used)
    limit = None if is_unlimited else float(limit_value)

    if disabled:
        limit = float(limit_value)

    remaining = None
    if limit is not None:
        remaining = max(limit - used_value, 0.0)

    return UsageMetric(
        used=used_value,
        limit=limit,
        remaining=remaining,
        is_unlimited=is_unlimited,
        reset_at=reset_at,
        disabled=disabled,
        unit=unit,
    )


def _serialize_subscription(
    subscription,
    tier: SubscriptionTier,
) -> SubscriptionSummary:
    """Convert a BillingSubscription (or None) into a serializable summary."""
    if subscription:
        return SubscriptionSummary(
            tier=subscription.tier.value,
            status=subscription.status.value,
            current_period_end=subscription.current_period_end,
            cancel_at_period_end=subscription.cancel_at_period_end,
        )

    # No subscription record (self-hosted or free default)
    fallback_status = SubscriptionStatus.ACTIVE.value
    return SubscriptionSummary(
        tier=tier.value,
        status=fallback_status,
        current_period_end=None,
        cancel_at_period_end=False,
    )


def _build_usage_breakdown(
    *,
    limits: TierLimits,
    chat_messages_used: int,
    runs_used: int,
    workflows_used: int,
    llm_credits_used: Decimal,
    reset_at: datetime,
) -> UsageBreakdown:
    """Assemble the usage payload for all metered resources."""
    return UsageBreakdown(
        chat_messages=_build_usage_metric(
            used=chat_messages_used,
            limit_value=limits.chat_messages_total,
            is_unlimited=limits.has_unlimited_chat,
            disabled=limits.is_chat_disabled,
            unit="messages",
        ),
        workflow_runs=_build_usage_metric(
            used=runs_used,
            limit_value=limits.runs_monthly,
            is_unlimited=limits.has_unlimited_runs,
            reset_at=reset_at,
            unit="runs",
        ),
        llm_credits=_build_usage_metric(
            used=llm_credits_used,
            limit_value=limits.llm_credits_monthly,
            is_unlimited=limits.has_unlimited_credits,
            reset_at=reset_at,
            unit="usd",
        ),
        workflows=_build_usage_metric(
            used=workflows_used,
            limit_value=limits.workflows,
            is_unlimited=limits.has_unlimited_workflows,
            unit="workflows",
        ),
    )


@router.get("", response_model=UsageResponse)
async def get_usage_summary(request: Request) -> UsageResponse:
    """
    Return the current user's usage and limits across all gated resources.
    """
    user = _require_user(request)
    limits = await get_limits_for_user(user)
    tier = await resolve_user_tier(user)
    subscription = await get_subscription_for_user(user)

    chat_messages_used = await get_total_chat_message_count(user)
    monthly_runs_used = await get_monthly_run_count(user)
    workflows_used = await get_workflow_count(user)
    llm_credits_used = await get_monthly_llm_credits_used(user)

    _, reset_at = await get_billing_period_for_user(user, subscription)
    usage_breakdown = _build_usage_breakdown(
        limits=limits,
        chat_messages_used=chat_messages_used,
        runs_used=monthly_runs_used,
        workflows_used=workflows_used,
        llm_credits_used=llm_credits_used,
        reset_at=reset_at,
    )

    return UsageResponse(
        tier=tier.value,
        is_self_hosted=config.is_self_hosted,
        subscription=_serialize_subscription(subscription, tier),
        limits={
            "poll_min_interval_seconds": limits.poll_min_interval_seconds,
        },
        usage=usage_breakdown,
    )


# =============================================================================
# Analytics Response Models
# =============================================================================


class ModelUsageItem(BaseModel):
    model: str
    provider: str | None
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    call_count: int


class OperationUsageItem(BaseModel):
    operation: str | None
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    call_count: int


class DailyUsageItem(BaseModel):
    date: str
    total_cost: float
    total_tokens: int
    call_count: int


class WorkflowUsageItem(BaseModel):
    workflow_id: str
    workflow_name: str | None
    total_cost: float
    total_tokens: int
    call_count: int


class UsageRecordItem(BaseModel):
    id: int
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    operation: str | None
    workflow_run_id: str | None
    created_at: datetime


class AnalyticsOverviewResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    total_cost: float
    total_tokens: int
    total_calls: int
    by_model: list[ModelUsageItem]
    by_operation: list[OperationUsageItem]
    daily_trend: list[DailyUsageItem]


class WorkflowCostsResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    workflows: list[WorkflowUsageItem]


class UsageRecordsResponse(BaseModel):
    records: list[UsageRecordItem]
    total: int
    limit: int
    offset: int


# =============================================================================
# Analytics Helpers
# =============================================================================


async def _resolve_period(
    user: User,
    start: Optional[datetime],
    end: Optional[datetime],
) -> tuple[datetime, datetime]:
    """Default start/end to the user's current billing period if not provided."""
    if start and end:
        return start, end
    period_start, period_end = await get_billing_period_for_user(user)
    return start or period_start, end or period_end


# =============================================================================
# Analytics Endpoints
# =============================================================================


@router.get("/analytics", response_model=AnalyticsOverviewResponse)
async def get_usage_analytics(
    request: Request,
    start: Optional[datetime] = Query(default=None, description="Period start (defaults to billing period)"),
    end: Optional[datetime] = Query(default=None, description="Period end (defaults to billing period)"),
    model: Optional[str] = Query(default=None, description="Filter by model name"),
    operation: Optional[str] = Query(default=None, description="Filter by operation type"),
) -> AnalyticsOverviewResponse:
    """Overview dashboard with breakdowns by model, operation, and daily trend."""
    user = _require_user(request)
    period_start, period_end = await _resolve_period(user, start, end)

    filter_kwargs = {"model": model, "operation": operation}

    by_model, by_operation, daily_trend = await asyncio.gather(
        get_llm_usage_by_model(user, period_start, period_end, **filter_kwargs),
        get_llm_usage_by_operation(user, period_start, period_end, **filter_kwargs),
        get_llm_usage_daily_trend(user, period_start, period_end, **filter_kwargs),
    )

    total_cost = sum(float(r["total_cost"]) for r in by_model) if by_model else 0.0
    total_tokens = sum(r["total_tokens"] for r in by_model) if by_model else 0
    total_calls = sum(r["call_count"] for r in by_model) if by_model else 0

    return AnalyticsOverviewResponse(
        period_start=period_start,
        period_end=period_end,
        total_cost=total_cost,
        total_tokens=total_tokens,
        total_calls=total_calls,
        by_model=[ModelUsageItem(**r) for r in by_model],
        by_operation=[OperationUsageItem(**r) for r in by_operation],
        daily_trend=[DailyUsageItem(**r) for r in daily_trend],
    )


@router.get("/analytics/workflows", response_model=WorkflowCostsResponse)
async def get_workflow_costs(
    request: Request,
    start: Optional[datetime] = Query(default=None, description="Period start (defaults to billing period)"),
    end: Optional[datetime] = Query(default=None, description="Period end (defaults to billing period)"),
) -> WorkflowCostsResponse:
    """Per-workflow cost aggregation (grouped by workflow, not individual runs)."""
    user = _require_user(request)
    period_start, period_end = await _resolve_period(user, start, end)

    workflows = await get_llm_usage_by_workflow(user, period_start, period_end)

    return WorkflowCostsResponse(
        period_start=period_start,
        period_end=period_end,
        workflows=[WorkflowUsageItem(**w) for w in workflows],
    )


@router.get("/analytics/records", response_model=UsageRecordsResponse)
async def get_usage_records(
    request: Request,
    start: Optional[datetime] = Query(default=None, description="Period start (defaults to billing period)"),
    end: Optional[datetime] = Query(default=None, description="Period end (defaults to billing period)"),
    limit: int = Query(default=50, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> UsageRecordsResponse:
    """Paginated detail records for audit/drill-down."""
    user = _require_user(request)
    period_start, period_end = await _resolve_period(user, start, end)

    records, total = await get_llm_usage_records_paginated(
        user, period_start, period_end, limit=limit, offset=offset,
    )

    return UsageRecordsResponse(
        records=[UsageRecordItem(**r) for r in records],
        total=total,
        limit=limit,
        offset=offset,
    )
