"""
Overage (usage-based pricing) API endpoints.

Provides endpoints for:
- Getting overage settings and current usage
- Enabling/disabling usage-based pricing
- Updating spending cap
- Viewing detailed overage usage records
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from seer.config import config
from seer.database.models import User
from seer.database.organization_models import Organization
from seer.database.overage_models import OverageRecordStatus, OverageUsageRecord
from seer.logger import get_logger
from seer.observability.constants import tiered_usage_limits

from seer.api.subscriptions.overage_service import (
    disable_overage,
    enable_overage,
    get_or_create_overage_settings,
    get_overage_usage_summary,
    is_overage_eligible,
    update_spending_cap,
)
from seer.api.subscriptions.stripe_service import get_org_subscription

logger = get_logger("api.overage.router")

router = APIRouter(prefix="/overage", tags=["overage"])


def _require_user(request: Request) -> User:
    """Extract authenticated user from request or raise 401."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _require_organization(request: Request) -> Organization:
    """Extract organization from request or raise 401."""
    org = getattr(request.state, "organization", None)
    if org is None:
        raise HTTPException(status_code=401, detail="Organization context required")
    return org


# --- Request/Response Models ---


class OverageSettingsResponse(BaseModel):
    """Response containing overage settings and usage."""

    enabled: bool
    eligible: bool
    spending_cap_cents: int
    spending_cap_dollars: float
    current_usage_cents: int
    current_usage_dollars: float
    remaining_cents: int
    remaining_dollars: float
    cap_reached: bool
    margin_multiplier: float
    margin_percent: float
    period_start: Optional[str] = None
    enabled_at: Optional[str] = None
    records: Optional[dict] = None


class EnableOverageRequest(BaseModel):
    """Request body for enabling usage-based pricing."""

    spending_cap_cents: int = Field(
        default=5000,
        ge=500,  # $5 minimum
        le=100000,  # $1000 maximum
        description="Spending cap in cents ($5 - $1000)",
    )


class EnableOverageResponse(BaseModel):
    """Response after enabling usage-based pricing."""

    success: bool
    settings: OverageSettingsResponse


class DisableOverageResponse(BaseModel):
    """Response after disabling usage-based pricing."""

    success: bool
    pending_charges_cents: int
    message: str


class UpdateCapRequest(BaseModel):
    """Request body for updating spending cap."""

    spending_cap_cents: int = Field(
        ge=500,
        le=100000,
        description="New spending cap in cents ($5 - $1000)",
    )


class UpdateCapResponse(BaseModel):
    """Response after updating spending cap."""

    success: bool
    spending_cap_cents: int
    spending_cap_dollars: float


class OverageUsageRecordResponse(BaseModel):
    """Individual overage usage record."""

    id: int
    base_cost_cents: int
    billed_amount_cents: int
    status: str
    created_at: str
    reported_to_stripe_at: Optional[str] = None


class OverageUsageListResponse(BaseModel):
    """Paginated list of overage usage records."""

    items: list[OverageUsageRecordResponse]
    total_count: int
    page: int
    page_size: int
    has_more: bool


class OverageConfigResponse(BaseModel):
    """Response containing overage configuration limits."""

    min_cap_cents: int
    max_cap_cents: int
    default_cap_cents: int
    default_margin_multiplier: float
    warning_threshold: float


# --- Endpoints ---


@router.get("/config", response_model=OverageConfigResponse)
async def get_overage_config():
    """
    Get overage configuration limits.

    Returns the minimum, maximum, and default values for overage settings.
    This is a public endpoint (no auth required).
    """
    return OverageConfigResponse(
        min_cap_cents=tiered_usage_limits.OVERAGE_MIN_CAP_CENTS,
        max_cap_cents=tiered_usage_limits.OVERAGE_MAX_CAP_CENTS,
        default_cap_cents=tiered_usage_limits.OVERAGE_DEFAULT_CAP_CENTS,
        default_margin_multiplier=tiered_usage_limits.OVERAGE_DEFAULT_MARGIN_MULTIPLIER,
        warning_threshold=tiered_usage_limits.OVERAGE_WARNING_THRESHOLD,
    )


@router.get("", response_model=OverageSettingsResponse)
async def get_overage_settings(request: Request):
    """
    Get current overage settings and usage.

    Returns:
    - Whether overage is enabled
    - Whether user is eligible for overage
    - Current spending cap
    - Current usage in this billing period
    - Remaining cap
    - Usage record counts by status
    """
    organization = _require_organization(request)

    # Check eligibility
    subscription = await get_org_subscription(organization)
    eligible = await is_overage_eligible(subscription)

    # Get or create overage settings
    settings = await get_or_create_overage_settings(organization)

    # Get usage summary
    summary = await get_overage_usage_summary(settings)

    return OverageSettingsResponse(
        enabled=summary["enabled"],
        eligible=eligible,
        spending_cap_cents=summary["spending_cap_cents"],
        spending_cap_dollars=summary["spending_cap_dollars"],
        current_usage_cents=summary["current_usage_cents"],
        current_usage_dollars=summary["current_usage_dollars"],
        remaining_cents=summary["remaining_cents"],
        remaining_dollars=summary["remaining_dollars"],
        cap_reached=summary["cap_reached"],
        margin_multiplier=summary["margin_multiplier"],
        margin_percent=(summary["margin_multiplier"] - 1) * 100,
        period_start=summary["period_start"],
        enabled_at=summary["enabled_at"],
        records=summary["records"],
    )


@router.post("/enable", response_model=EnableOverageResponse)
async def enable_overage_pricing(request: Request, body: EnableOverageRequest):
    """
    Enable usage-based pricing for LLM credits.

    Requirements:
    - Paid tier subscription (LITE or PRO)
    - Active subscription status
    - Payment method on file

    After enabling, usage beyond subscription allowance will be billed
    at pass-through LLM cost plus margin (default 30%).
    """
    if not config.is_stripe_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    organization = _require_organization(request)

    # Get subscription
    subscription = await get_org_subscription(organization)

    # Check eligibility
    if not await is_overage_eligible(subscription):
        raise HTTPException(
            status_code=400,
            detail="Overage pricing is only available for paid tier subscriptions with a payment method on file.",
        )

    try:
        settings = await enable_overage(
            organization=organization,
            subscription=subscription,
            spending_cap_cents=body.spending_cap_cents,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Get updated summary
    summary = await get_overage_usage_summary(settings)

    return EnableOverageResponse(
        success=True,
        settings=OverageSettingsResponse(
            enabled=summary["enabled"],
            eligible=True,
            spending_cap_cents=summary["spending_cap_cents"],
            spending_cap_dollars=summary["spending_cap_dollars"],
            current_usage_cents=summary["current_usage_cents"],
            current_usage_dollars=summary["current_usage_dollars"],
            remaining_cents=summary["remaining_cents"],
            remaining_dollars=summary["remaining_dollars"],
            cap_reached=summary["cap_reached"],
            margin_multiplier=summary["margin_multiplier"],
            margin_percent=(summary["margin_multiplier"] - 1) * 100,
            period_start=summary["period_start"],
            enabled_at=summary["enabled_at"],
            records=summary["records"],
        ),
    )


@router.post("/disable", response_model=DisableOverageResponse)
async def disable_overage_pricing(request: Request):
    """
    Disable usage-based pricing.

    Note: Any pending charges in the current billing period will still
    be billed. Only new overages will be blocked after disabling.
    """
    organization = _require_organization(request)

    # Get subscription
    subscription = await get_org_subscription(organization)

    settings = await disable_overage(
        organization=organization,
        subscription=subscription,
    )

    pending_charges = settings.current_period_overage_cents

    message = "Usage-based pricing has been disabled."
    if pending_charges > 0:
        message += f" ${pending_charges / 100:.2f} in pending charges will still be billed."

    return DisableOverageResponse(
        success=True,
        pending_charges_cents=pending_charges,
        message=message,
    )


@router.put("/cap", response_model=UpdateCapResponse)
async def update_overage_cap(request: Request, body: UpdateCapRequest):
    """
    Update the spending cap for usage-based pricing.

    The cap can be updated whether overage is currently enabled or not.
    """
    organization = _require_organization(request)

    try:
        settings = await update_spending_cap(
            organization=organization,
            spending_cap_cents=body.spending_cap_cents,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UpdateCapResponse(
        success=True,
        spending_cap_cents=settings.spending_cap_cents,
        spending_cap_dollars=float(settings.spending_cap_dollars),
    )


@router.get("/usage", response_model=OverageUsageListResponse)
async def list_overage_usage(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status: pending, reported, failed"),
):
    """
    List detailed overage usage records for the current billing period.

    Returns individual usage records with their Stripe reporting status.
    """
    organization = _require_organization(request)

    # Get overage settings
    settings = await get_or_create_overage_settings(organization)

    # Build query
    query = OverageUsageRecord.filter(overage_settings=settings)

    # Apply status filter if provided
    if status:
        try:
            status_enum = OverageRecordStatus(status)
            query = query.filter(status=status_enum)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: pending, reported, failed",
            ) from exc

    # Get total count
    total_count = await query.count()

    # Get paginated records
    offset = (page - 1) * page_size
    records = await query.order_by("-created_at").offset(offset).limit(page_size + 1)

    # Check if there are more records
    has_more = len(records) > page_size
    records = records[:page_size]

    items = [
        OverageUsageRecordResponse(
            id=record.id,
            base_cost_cents=record.base_cost_cents,
            billed_amount_cents=record.billed_amount_cents,
            status=record.status.value,
            created_at=record.created_at.isoformat(),
            reported_to_stripe_at=(
                record.reported_to_stripe_at.isoformat()
                if record.reported_to_stripe_at
                else None
            ),
        )
        for record in records
    ]

    return OverageUsageListResponse(
        items=items,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_more=has_more,
    )
