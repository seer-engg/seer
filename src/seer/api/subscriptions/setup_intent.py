"""
Setup Intent API endpoints for payment method collection during onboarding.

Provides endpoints for:
- Creating Setup Intents for collecting payment methods
- Confirming Setup Intent success
- Checking payment method status
"""
from datetime import datetime, timezone

import stripe
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from seer.config import config
from seer.database.models import User
from seer.database.subscription_models import BillingProfile
from seer.logger import get_logger

from .stripe_service import get_or_create_stripe_customer

logger = get_logger("api.subscriptions.setup_intent")

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


def _require_user(request: Request) -> User:
    """Extract authenticated user from request or raise 401."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# --- Request/Response Models ---


class SetupIntentResponse(BaseModel):
    """Response containing Setup Intent client secret."""
    client_secret: str
    stripe_customer_id: str


class ConfirmSetupRequest(BaseModel):
    """Request body for confirming Setup Intent."""
    setup_intent_id: str


class SetupConfirmResponse(BaseModel):
    """Response for Setup Intent confirmation."""
    success: bool
    message: str


class PaymentMethodStatusResponse(BaseModel):
    """Response for payment method status check."""
    has_payment_method: bool
    payment_method_added_at: str | None = None


# --- Endpoints ---


@router.post("/setup-intent", response_model=SetupIntentResponse)
async def create_setup_intent(request: Request):
    """
    Create a Stripe Setup Intent for collecting payment method during onboarding.

    This allows collecting payment information without charging the user.
    The payment method will be saved for future use after trial expires.

    Returns:
        SetupIntentResponse with client_secret for Stripe Elements integration
    """
    if not config.is_stripe_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    user = _require_user(request)

    try:
        # Get or create Stripe customer
        customer_id = await get_or_create_stripe_customer(user)

        # Create Setup Intent with off_session usage for future charges
        setup_intent = stripe.SetupIntent.create(
            customer=customer_id,
            payment_method_types=["card"],
            usage="off_session",  # Allow charging without customer present
            metadata={
                "user_id": user.user_id,  # Clerk user ID
                "seer_user_id": str(user.id),
            }
        )

        logger.info(
            "Created Setup Intent %s for user %s (customer %s)",
            setup_intent.id, user.user_id, customer_id
        )

        return SetupIntentResponse(
            client_secret=setup_intent.client_secret,
            stripe_customer_id=customer_id,
        )

    except stripe.error.StripeError as exc:
        logger.error("Stripe Setup Intent creation error for user %s: %s", user.user_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/setup-intent/confirm", response_model=SetupConfirmResponse)
async def confirm_setup_intent(request: Request, body: ConfirmSetupRequest):
    """
    Confirm that Setup Intent succeeded and update user's payment method status.

    This endpoint is called by the frontend after Stripe confirms the payment method.
    It verifies the Setup Intent status and updates the BillingProfile accordingly.

    Args:
        body: Contains the setup_intent_id to verify

    Returns:
        SetupConfirmResponse indicating success
    """
    if not config.is_stripe_configured:
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    user = _require_user(request)

    try:
        # Retrieve Setup Intent from Stripe to verify status
        setup_intent = stripe.SetupIntent.retrieve(body.setup_intent_id)

        if setup_intent.status != "succeeded":
            logger.warning(
                "Setup Intent %s for user %s has status %s (expected 'succeeded')",
                body.setup_intent_id, user.user_id, setup_intent.status
            )
            raise HTTPException(
                status_code=400,
                detail=f"Setup Intent status is {setup_intent.status}, expected 'succeeded'"
            )

        # Get billing profile and update payment method status
        billing_profile = await BillingProfile.get_or_none(owner_user=user)
        if not billing_profile:
            logger.error("No billing profile found for user %s during Setup Intent confirmation", user.user_id)
            raise HTTPException(status_code=404, detail="Billing profile not found")

        billing_profile.has_payment_method = True
        billing_profile.payment_method_added_at = datetime.now(timezone.utc)
        await billing_profile.save(update_fields=["has_payment_method", "payment_method_added_at"])

        logger.info(
            "Confirmed Setup Intent %s for user %s, updated payment method status",
            body.setup_intent_id, user.user_id
        )

        return SetupConfirmResponse(
            success=True,
            message="Payment method successfully added"
        )

    except stripe.error.StripeError as exc:
        logger.error("Stripe Setup Intent retrieval error: %s", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/payment-method/status", response_model=PaymentMethodStatusResponse)
async def get_payment_method_status(request: Request):
    """
    Get payment method status for the authenticated user.

    Returns whether the user has a payment method on file and when it was added.
    """
    user = _require_user(request)

    billing_profile = await BillingProfile.get_or_none(owner_user=user)

    if not billing_profile:
        return PaymentMethodStatusResponse(
            has_payment_method=False,
            payment_method_added_at=None
        )

    return PaymentMethodStatusResponse(
        has_payment_method=billing_profile.has_payment_method,
        payment_method_added_at=(
            billing_profile.payment_method_added_at.isoformat()
            if billing_profile.payment_method_added_at
            else None
        )
    )
