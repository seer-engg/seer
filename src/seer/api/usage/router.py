"""
Usage summary endpoints for exposing current limits and consumption.

Provides a consolidated view of usage across workflows, runs, chat messages,
and LLM credits so the frontend can display upgrade nudges and remaining quota.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from seer.config import config
from seer.database import User
from seer.database.subscription_models import SubscriptionStatus, SubscriptionTier
from seer.observability import (
    get_limits_for_user,
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
