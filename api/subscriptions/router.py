"""
Subscription management API endpoints.

Provides endpoints for:
- Getting current subscription status
- Creating Stripe Checkout sessions
- Creating Stripe Customer Portal sessions
- Handling Stripe webhooks
"""
from typing import Optional

import stripe
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel
from tortoise.exceptions import IntegrityError

from shared.config import config
from shared.database.models import User
from shared.database.subscription_models import (
    StripeWebhookEvent,
    StripeWebhookEventStatus,
)
from shared.logger import get_logger

from .clerk_sync import sync_stripe_customer_to_clerk
from .stripe_service import (
    create_checkout_session,
    create_portal_session,
    get_user_subscription,
    verify_webhook_signature,
)
from worker.tasks.stripe import process_stripe_webhook_event

logger = get_logger("api.subscriptions.router")

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


def _require_user(request: Request) -> User:
    """Extract authenticated user from request or raise 401."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# --- Request/Response Models ---


class CheckoutRequest(BaseModel):
    """Request body for creating a checkout session."""
    price_id: str


class CheckoutResponse(BaseModel):
    """Response containing checkout URL."""
    checkout_url: str


class PortalResponse(BaseModel):
    """Response containing portal URL."""
    portal_url: str


class SubscriptionResponse(BaseModel):
    """Response containing subscription details."""
    tier: str
    status: str
    current_period_end: Optional[str] = None
    cancel_at_period_end: bool = False


class PriceInfo(BaseModel):
    """Price information for a billing cycle."""
    price: int
    price_id: Optional[str] = None


class TierPricing(BaseModel):
    """Pricing information for a subscription tier."""
    tier: str
    name: str
    monthly: PriceInfo
    annual: PriceInfo


class PricingResponse(BaseModel):
    """Response containing all subscription pricing."""
    prices: list[TierPricing]


# --- Endpoints ---


@router.get("/pricing", response_model=PricingResponse)
async def get_pricing():
    """
    Get available subscription prices.

    Returns pricing information for all subscription tiers with
    monthly and annual options.
    """
    return PricingResponse(prices=[
        TierPricing(
            tier="pro",
            name="Pro",
            monthly=PriceInfo(price=20, price_id=config.stripe_price_pro_monthly),
            annual=PriceInfo(price=200, price_id=config.stripe_price_pro_annual),
        ),
        TierPricing(
            tier="pro_plus",
            name="Pro+",
            monthly=PriceInfo(price=60, price_id=config.stripe_price_proplus_monthly),
            annual=PriceInfo(price=600, price_id=config.stripe_price_proplus_annual),
        ),
        TierPricing(
            tier="ultra",
            name="Ultra",
            monthly=PriceInfo(price=100, price_id=config.stripe_price_ultra_monthly),
            annual=PriceInfo(price=1000, price_id=config.stripe_price_ultra_annual),
        ),
    ])


@router.get("/current", response_model=SubscriptionResponse)
async def get_current_subscription(request: Request):
    """
    Get current user's subscription status.

    Returns the user's current tier, status, and billing period information.
    """
    user = _require_user(request)
    subscription = await get_user_subscription(user)

    return SubscriptionResponse(
        tier=subscription.tier.value,
        status=subscription.status.value,
        current_period_end=(
            subscription.current_period_end.isoformat()
            if subscription.current_period_end
            else None
        ),
        cancel_at_period_end=subscription.cancel_at_period_end,
    )


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(request: Request, body: CheckoutRequest):
    """
    Create Stripe Checkout session for subscription.

    Creates a hosted checkout page for the user to complete payment.
    On success, redirects to billing settings with success message.
    """
    if not config.is_stripe_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    user = _require_user(request)

    success_url = f"{config.frontend_url}/settings/billing?success=true"
    cancel_url = f"{config.frontend_url}/settings/billing?canceled=true"

    try:
        checkout_url = await create_checkout_session(
            user=user,
            price_id=body.price_id,
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return CheckoutResponse(checkout_url=checkout_url)
    except stripe.error.StripeError as e:
        logger.error("Stripe checkout error for user %s: %s", user.user_id, str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/portal", response_model=PortalResponse)
async def create_portal(request: Request):
    """
    Create Stripe Customer Portal session.

    Creates a session for the user to manage their subscription,
    update payment methods, view invoices, and cancel.
    """
    if not config.is_stripe_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    user = _require_user(request)
    return_url = f"{config.frontend_url}/settings/billing"

    try:
        portal_url = await create_portal_session(user, return_url)
        return PortalResponse(portal_url=portal_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except stripe.error.StripeError as e:
        logger.error("Stripe portal error for user %s: %s", user.user_id, str(e))
        raise HTTPException(status_code=400, detail=str(e))


# --- Webhook Handler ---


@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(alias="Stripe-Signature"),
):
    """
    Handle Stripe webhook events.

    Processes subscription lifecycle events to keep our database
    in sync with Stripe's subscription state.

    Events handled:
    - checkout.session.completed: Sync customer ID to Clerk
    - customer.subscription.created/updated: Sync subscription state
    - customer.subscription.deleted: Revert to free tier
    - invoice.payment_failed: Log for notification (future)
    """
    if not config.is_stripe_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    payload = await request.body()

    try:
        event = verify_webhook_signature(payload, stripe_signature)
    except stripe.error.SignatureVerificationError:
        logger.warning("Invalid Stripe webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_dict = event.to_dict_recursive() if hasattr(event, "to_dict_recursive") else event
    event_id = event_dict.get("id")
    event_type = event_dict.get("type")

    if not event_id or not event_type:
        raise HTTPException(status_code=400, detail="Missing event id or type")

    logger.info("Persisting Stripe webhook %s (%s)", event_id, event_type)

    record: StripeWebhookEvent | None = None
    try:
        record = await StripeWebhookEvent.create(
            event_id=event_id,
            type=event_type,
            payload=event_dict,
            status=StripeWebhookEventStatus.RECEIVED,
        )
    except IntegrityError:
        record = await StripeWebhookEvent.get_or_none(event_id=event_id)
        if record and record.status == StripeWebhookEventStatus.PROCESSED:
            logger.info("Stripe event %s already processed; acknowledging", event_id)
            return {"status": "ok"}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to persist Stripe webhook %s: %s", event_id, exc)
        raise HTTPException(status_code=500, detail="Failed to persist webhook")

    if not record:
        logger.error("Could not load Stripe webhook record for %s", event_id)
        raise HTTPException(status_code=500, detail="Failed to persist webhook")

    try:
        await process_stripe_webhook_event.kiq(event_db_id=record.id)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to enqueue Stripe webhook %s: %s", event_id, exc)
        raise HTTPException(status_code=500, detail="Failed to enqueue webhook")

    return {"status": "queued"}
