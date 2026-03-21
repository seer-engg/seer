# pylint: disable=broad-exception-caught,import-outside-toplevel,cyclic-import,too-many-lines
# Reason: Stripe operations require broad error handling; webhook controller imported lazily to avoid cycles during app init
"""
Stripe service layer for subscription management.

Handles Stripe customer creation, checkout sessions, portal sessions,
and subscription state synchronization from webhooks.
"""
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Tuple, Union

import stripe
from tortoise.backends.base.client import BaseDBAsyncClient
from tortoise.transactions import in_transaction

from seer.config import config
from seer.database.models import User
from seer.database.subscription_models import (
    BillingSubscription,
    StripeCustomer,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.database.organization_models import Organization, OrganizationType
from seer.database.overage_models import OverageSettings
from seer.logger import get_logger
from seer.api.subscriptions.pricing_catalog import get_price_id_to_tier_map

logger = get_logger("api.subscriptions.stripe_service")

# Initialize Stripe with API key
if config.stripe_secret_key:
    stripe.api_key = config.stripe_secret_key


def _build_price_to_tier_map() -> dict[str, SubscriptionTier]:
    """Build mapping from Stripe price IDs to subscription tiers using cached pricing catalog."""
    mapping: dict[str, SubscriptionTier] = {}
    try:
        raw_map = get_price_id_to_tier_map()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load pricing catalog for tier mapping: %s", exc)
        return mapping

    for price_id, tier_str in raw_map.items():
        try:
            mapping[price_id] = SubscriptionTier(tier_str)
        except ValueError:
            logger.warning("Unknown tier '%s' for price %s — skipping", tier_str, price_id)
    return mapping


def _timestamp_to_datetime(timestamp: Any) -> Optional[datetime]:
    """Convert a Stripe timestamp to aware datetime or return None."""
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def _timestamp_to_iso(timestamp: Any) -> Optional[str]:
    """Convert a Stripe timestamp to ISO string or return None."""
    dt = _timestamp_to_datetime(timestamp)
    return dt.isoformat() if dt else None


def _paginate_stripe_list(
    list_fn: Callable[..., Any],
    *,
    page: int,
    page_size: int,
    **kwargs: Any,
) -> Tuple[list[dict], bool]:
    """
    Emulate numbered pagination over Stripe cursor-based lists.

    Args:
        list_fn: Callable that accepts limit/starting_after and returns a Stripe list response.
        page: 1-based page number.
        page_size: Number of records per page (capped at 100 by Stripe).
        **kwargs: Extra parameters forwarded to the Stripe list call.

    Returns:
        (items, has_more) tuple for the requested page.

    Raises:
        ValueError: When page or page_size are invalid.
    """
    if page < 1:
        raise ValueError("page must be >= 1")
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size must be between 1 and 100")

    starting_after = None
    to_skip = (page - 1) * page_size

    # Walk pages until we reach the desired offset.
    while to_skip > 0:
        batch_limit = min(100, to_skip)
        response = list_fn(limit=batch_limit, starting_after=starting_after, **kwargs)
        batch = response.get("data", [])
        if not batch:
            return [], False
        to_skip -= len(batch)
        starting_after = batch[-1].get("id")
        if not response.get("has_more") and to_skip > 0:
            return [], False

    response = list_fn(limit=page_size, starting_after=starting_after, **kwargs)
    items = response.get("data", [])
    has_more = bool(response.get("has_more"))
    return items, has_more


def _maybe_fetch_subscription(stripe_subscription: Union[dict, str, stripe.Subscription]) -> Optional[stripe.Subscription]:
    """
    Ensure we have a full subscription object (with period dates and items).

    Some webhook payloads (or mocked events) may omit fields like current_period_start.
    In those cases, fetch the subscription from Stripe to avoid KeyErrors.
    """
    try:
        subscription_id = stripe_subscription if isinstance(stripe_subscription, str) else stripe_subscription.get("id")
    except AttributeError:
        subscription_id = None

    needs_fetch = isinstance(stripe_subscription, str)
    if not needs_fetch and hasattr(stripe_subscription, "get"):
        items = stripe_subscription.get("items", {}).get("data", [])
        missing_periods = (
            stripe_subscription.get("current_period_start") is None
            or stripe_subscription.get("current_period_end") is None
        )
        needs_fetch = missing_periods or not items

    if needs_fetch and subscription_id:
        try:
            return stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
        except stripe.error.StripeError as exc:
            logger.error("Failed to fetch Stripe subscription %s: %s", subscription_id, exc)
            return None

    return stripe_subscription  # type: ignore[return-value]


# Stripe status to our status mapping
STRIPE_STATUS_MAP = {
    "active": SubscriptionStatus.ACTIVE,
    "canceled": SubscriptionStatus.CANCELED,
    "past_due": SubscriptionStatus.PAST_DUE,
    "trialing": SubscriptionStatus.TRIALING,
    "incomplete": SubscriptionStatus.INCOMPLETE,
    "incomplete_expired": SubscriptionStatus.CANCELED,
    "unpaid": SubscriptionStatus.PAST_DUE,
}


def _resolve_subscription_tier(subscription_obj: dict, current_tier: SubscriptionTier) -> SubscriptionTier:
    """
    Resolve subscription tier from Stripe subscription items.

    Tries multiple resolution strategies in order:
    1. Lookup in cached price-to-tier map
    2. Fetch price directly from Stripe and read metadata
    3. Fall back to current tier

    Args:
        subscription_obj: Stripe subscription object
        current_tier: Current tier to use as fallback

    Returns:
        Resolved subscription tier
    """
    items = subscription_obj.get("items", {}).get("data", [])
    if not items:
        return current_tier

    price_id = items[0].get("price", {}).get("id")
    if not price_id:
        return current_tier

    # Try cached price-to-tier map first
    price_to_tier = _build_price_to_tier_map()
    tier = price_to_tier.get(price_id)

    # Fallback: fetch price directly from Stripe and read metadata.tier
    if tier is None:
        try:
            fetched_price = stripe.Price.retrieve(price_id)
            tier_str = (fetched_price.get("metadata") or {}).get("tier")
            if tier_str:
                tier = SubscriptionTier(tier_str)
        except (stripe.error.StripeError, ValueError) as exc:
            logger.warning("Failed to resolve tier for price %s: %s", price_id, exc)

    return tier if tier is not None else current_tier


def _update_subscription_fields(
    subscription: BillingSubscription,
    stripe_obj: dict,
    tier: SubscriptionTier,
) -> None:
    """
    Update all fields on a BillingSubscription from Stripe data.

    Args:
        subscription: The subscription to update
        stripe_obj: Stripe subscription object
        tier: Resolved tier for the subscription
    """
    status = stripe_obj.get("status")
    mapped_status = STRIPE_STATUS_MAP.get(status, subscription.status)

    subscription.stripe_subscription_id = stripe_obj.get("id")
    subscription.tier = tier
    subscription.status = mapped_status

    current_period_start_ts = stripe_obj.get("current_period_start")
    current_period_end_ts = stripe_obj.get("current_period_end")

    if current_period_start_ts is not None:
        subscription.current_period_start = _timestamp_to_datetime(current_period_start_ts)
    if current_period_end_ts is not None:
        subscription.current_period_end = _timestamp_to_datetime(current_period_end_ts)

    subscription.cancel_at_period_end = bool(stripe_obj.get("cancel_at_period_end", False))


async def sync_subscription_for_invoice(invoice: dict) -> Optional[BillingSubscription]:
    """
    Sync subscription based on invoice events (payment succeeded/failed).
    Uses V2 org-centric model.
    """
    subscription_id = invoice.get("subscription")
    if not subscription_id:
        logger.warning("Invoice %s missing subscription ID", invoice.get("id"))
        return None
    return await sync_subscription_from_stripe(subscription_id)


async def process_stripe_event(event_type: str | None, data: dict) -> None:
    """
    Deprecated: maintained for backward compatibility. Delegates to StripeWebhookController.
    """
    from seer.api.subscriptions.stripe_webhook_controller import stripe_webhook_controller  # imported here to avoid cycle

    await stripe_webhook_controller.process_event(event_type, data)


def verify_webhook_signature(payload: bytes, signature: str) -> dict:
    """
    Verify Stripe webhook signature and return the event.

    Args:
        payload: Raw request body bytes
        signature: Stripe-Signature header value

    Returns:
        The verified Stripe event object

    Raises:
        stripe.error.SignatureVerificationError: If signature is invalid
    """
    return stripe.Webhook.construct_event(
        payload,
        signature,
        config.stripe_webhook_secret,
    )


# ============================================================================
# Organization Billing Functions
# ============================================================================


async def get_org_subscription(organization: Organization) -> BillingSubscription:
    """
    Get organization's billing subscription or create with free tier default.

    Args:
        organization: The organization

    Returns:
        The organization's subscription record
    """
    subscription, created = await BillingSubscription.get_or_create(
        organization=organization,
        defaults={
            "tier": SubscriptionTier.FREE,
            "status": SubscriptionStatus.ACTIVE,
        }
    )
    subscription.organization = organization
    if created:
        logger.info("Created free tier subscription for organization %s", organization.id)
    return subscription


async def create_org_portal_session(organization: Organization, user: User, return_url: str) -> str:
    """
    Create a Stripe Customer Portal session for an organization.

    Args:
        organization: The organization
        user: The user accessing the portal (for customer creation if needed)
        return_url: URL to return to after portal session

    Returns:
        The Stripe Customer Portal URL
    """
    customer_id = await get_or_create_org_stripe_customer(organization, user)

    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=return_url,
    )

    logger.info("Created portal session for organization %s", organization.id)

    return session.url


async def create_org_checkout_session(
    organization: Organization,
    user: User,
    price_id: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe Checkout session for an organization and return the checkout URL.

    Used when an org needs to purchase a subscription.

    Args:
        organization: The organization to create checkout for
        user: The user initiating checkout (for customer creation if needed)
        price_id: Stripe Price ID for the subscription plan
        success_url: URL to redirect to on successful payment
        cancel_url: URL to redirect to if payment is canceled

    Returns:
        The Stripe Checkout session URL
    """
    customer_id = await get_or_create_org_stripe_customer(organization, user)

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        allow_promotion_codes=True,
        billing_address_collection="auto",
        metadata={
            "organization_id": str(organization.id),
            "organization_name": organization.name,
            "billing_type": "team" if organization.type == OrganizationType.TEAM else "personal",
        },
    )

    logger.info(
        "Created checkout session %s for organization %s, price %s",
        session.id, organization.id, price_id
    )

    return session.url


async def list_org_invoices(organization: Organization, page: int, page_size: int) -> dict:
    """
    List invoices for an organization's Stripe customer.

    Args:
        organization: The organization
        page: Page number (1-based)
        page_size: Items per page

    Returns:
        Dict with items list and has_more flag
    """
    if not organization.stripe_customer_id:
        return {"items": [], "has_more": False}

    stripe_customer = await StripeCustomer.get(id=organization.stripe_customer_id)

    items, has_more = _paginate_stripe_list(
        stripe.Invoice.list,
        page=page,
        page_size=page_size,
        customer=stripe_customer.stripe_customer_id,
        expand=["data.customer"],
    )

    def _serialize_invoice(invoice: dict) -> dict[str, Any]:
        return {
            "id": invoice.get("id"),
            "number": invoice.get("number"),
            "status": invoice.get("status"),
            "currency": invoice.get("currency"),
            "total": invoice.get("total"),
            "amount_paid": invoice.get("amount_paid"),
            "amount_due": invoice.get("amount_due"),
            "created_at": _timestamp_to_iso(invoice.get("created")),
            "period_start": _timestamp_to_iso(invoice.get("period_start")),
            "period_end": _timestamp_to_iso(invoice.get("period_end")),
            "hosted_invoice_url": invoice.get("hosted_invoice_url"),
            "invoice_pdf": invoice.get("invoice_pdf"),
            "billing_reason": invoice.get("billing_reason"),
        }

    return {
        "items": [_serialize_invoice(item) for item in items],
        "has_more": has_more,
    }


async def get_or_create_org_stripe_customer(organization: Organization, user: User) -> str:
    """
    Get existing Stripe customer for org or create a new one with StripeCustomer audit record.

    Args:
        organization: The organization
        user: The user creating/accessing the customer (for customer name/email and audit)

    Returns:
        The Stripe customer ID
    """
    # Check if org already has a stripe customer linked
    if organization.stripe_customer_id:
        stripe_customer = await StripeCustomer.get(id=organization.stripe_customer_id)
        return stripe_customer.stripe_customer_id

    # Lock the organization row to avoid creating duplicate customers
    async with in_transaction() as conn:
        locked_org = await Organization.select_for_update().using_db(conn).get(id=organization.id)

        # Re-check after acquiring lock
        if locked_org.stripe_customer_id:
            stripe_customer = await StripeCustomer.get(id=locked_org.stripe_customer_id)
            return stripe_customer.stripe_customer_id

        # Create Stripe customer
        customer_params = {
            "email": user.email,
            "name": organization.name,
            "metadata": {
                "organization_id": str(organization.id),
                "organization_name": organization.name,
                "created_by_user_id": str(user.id),
                "billing_type": "team" if organization.type == OrganizationType.TEAM else "personal",
            }
        }

        if config.env == "dev":
            test_clock = stripe.test_helpers.TestClock.create(
                frozen_time=int(time.time()),
                name=f"Test clock for org {organization.name}",
            )
            customer_params["test_clock"] = test_clock.id
            logger.info("Created test clock %s for org %s", test_clock.id, organization.id)

        customer = stripe.Customer.create(**customer_params)
        logger.info("Created Stripe customer %s for organization %s", customer.id, organization.id)

        # Create StripeCustomer record for audit trail
        stripe_customer = await StripeCustomer.create(
            stripe_customer_id=customer.id,
            created_by_user=user,
            using_db=conn,
        )

        # Link to organization
        locked_org.stripe_customer_id = stripe_customer.id
        await locked_org.save(update_fields=["stripe_customer_id"], using_db=conn)

        return customer.id


async def sync_subscription_from_stripe(
    stripe_subscription: Union[dict, str, stripe.Subscription],
) -> Optional[BillingSubscription]:
    """
    Sync subscription state from Stripe webhook data using V2 org-centric model.

    Looks up organization via StripeCustomer table.

    Args:
        stripe_subscription: The Stripe subscription object or subscription ID

    Returns:
        The updated BillingSubscription or None if customer not found
    """
    subscription_obj = _maybe_fetch_subscription(stripe_subscription)
    if not subscription_obj:
        return None

    customer_id = subscription_obj.get("customer")
    subscription_id = subscription_obj.get("id")

    if not customer_id or not subscription_id:
        logger.warning("Stripe subscription payload missing id/customer: %s", subscription_obj)
        return None

    # V2: Look up via StripeCustomer -> Organization
    stripe_customer = await StripeCustomer.get_or_none(stripe_customer_id=customer_id)
    if not stripe_customer:
        logger.warning(
            "No StripeCustomer found for Stripe customer %s, subscription %s",
            customer_id, subscription_id
        )
        return None

    organization = await Organization.get_or_none(stripe_customer_id=stripe_customer.id)
    if not organization:
        logger.warning(
            "No Organization found for StripeCustomer %s, subscription %s",
            stripe_customer.id, subscription_id
        )
        return None

    # Get or create subscription for this organization
    subscription, _ = await BillingSubscription.get_or_create(
        organization=organization,
        defaults={
            "tier": SubscriptionTier.FREE,
            "status": SubscriptionStatus.ACTIVE,
        },
    )
    subscription.organization = organization

    # Resolve tier from Stripe price metadata
    tier = _resolve_subscription_tier(subscription_obj, subscription.tier)

    # Update all subscription fields from Stripe data
    _update_subscription_fields(subscription, subscription_obj, tier)

    await subscription.save()

    logger.info(
        "Synced subscription for org %s (customer %s): tier=%s, status=%s",
        organization.id, customer_id, tier.value, subscription.status.value
    )

    return subscription


async def handle_subscription_deleted(stripe_subscription: dict) -> Optional[BillingSubscription]:
    """
    Handle subscription cancellation/deletion using V2 org-centric model.

    Reverts organization to free tier when subscription is deleted.

    Args:
        stripe_subscription: The Stripe subscription object from webhook

    Returns:
        The updated BillingSubscription or None if customer not found
    """
    customer_id = stripe_subscription.get("customer")
    if not customer_id:
        logger.warning("Subscription deletion payload missing customer: %s", stripe_subscription)
        return None

    # V2: Look up via StripeCustomer -> Organization
    stripe_customer = await StripeCustomer.get_or_none(stripe_customer_id=customer_id)
    if not stripe_customer:
        logger.warning(
            "No StripeCustomer found for Stripe customer %s on subscription deletion",
            customer_id
        )
        return None

    organization = await Organization.get_or_none(stripe_customer_id=stripe_customer.id)
    if not organization:
        logger.warning(
            "No Organization found for StripeCustomer %s on subscription deletion",
            stripe_customer.id
        )
        return None

    subscription, _ = await BillingSubscription.get_or_create(
        organization=organization,
        defaults={
            "tier": SubscriptionTier.FREE,
            "status": SubscriptionStatus.ACTIVE,
        },
    )

    # Revert to free tier
    subscription.tier = SubscriptionTier.FREE
    subscription.status = SubscriptionStatus.ACTIVE
    subscription.stripe_subscription_id = None
    subscription.current_period_start = None
    subscription.current_period_end = None
    subscription.cancel_at_period_end = False

    await subscription.save()

    logger.info("Reverted org %s to free tier after subscription deletion", organization.id)

    return subscription


async def transfer_subscription_between_orgs(
    source_org: Organization,
    target_org: Organization,
    conn: BaseDBAsyncClient | None = None,
) -> None:
    """
    Transfer billing from one organization to another.

    This moves the StripeCustomer assignment and subscription from source_org to target_org.
    Used when a user creates a team and wants to transfer their personal subscription.

    All database operations are wrapped in a transaction for atomicity.

    Args:
        source_org: The source organization (e.g., personal org with active subscription)
        target_org: The target organization (e.g., new team org)

    Raises:
        ValueError: If source org has no stripe customer or subscription
    """
    if not source_org.stripe_customer_id:
        raise ValueError(f"Source organization {source_org.id} has no Stripe customer")

    get_kwargs = {"using_db": conn} if conn is not None else {}

    # Get the source subscription
    source_sub = await BillingSubscription.get_or_none(organization=source_org, **get_kwargs)
    if not source_sub:
        raise ValueError(f"Source organization {source_org.id} has no subscription")

    stripe_customer = await StripeCustomer.get(id=source_org.stripe_customer_id, **get_kwargs)

    async def _transfer_in_db(tx_conn: BaseDBAsyncClient) -> None:
        # Lock both organizations
        locked_source = await Organization.select_for_update().using_db(tx_conn).get(id=source_org.id)
        locked_target = await Organization.select_for_update().using_db(tx_conn).get(id=target_org.id)

        # Transfer stripe_customer FK
        locked_target.stripe_customer_id = locked_source.stripe_customer_id
        locked_target.has_payment_method = locked_source.has_payment_method
        locked_target.payment_method_added_at = locked_source.payment_method_added_at
        await locked_target.save(
            update_fields=["stripe_customer_id", "has_payment_method", "payment_method_added_at"],
            using_db=tx_conn
        )

        # Clear source org's billing
        locked_source.stripe_customer_id = None
        locked_source.has_payment_method = False
        locked_source.payment_method_added_at = None
        await locked_source.save(
            update_fields=["stripe_customer_id", "has_payment_method", "payment_method_added_at"],
            using_db=tx_conn
        )

        # Delete any existing subscription on target org (e.g., FREE created by create_team_organization)
        await BillingSubscription.filter(organization=locked_target).using_db(tx_conn).delete()

        # Move subscription to target org
        source_sub.organization = locked_target
        await source_sub.save(update_fields=["organization_id"], using_db=tx_conn)

        # Transfer overage settings if exists
        source_overage = await OverageSettings.get_or_none(organization=source_org, using_db=tx_conn)
        if source_overage:
            source_overage.organization = locked_target
            await source_overage.save(update_fields=["organization_id"], using_db=tx_conn)

        # Create FREE tier subscription for source org
        await BillingSubscription.create(
            organization=locked_source,
            tier=SubscriptionTier.FREE,
            status=SubscriptionStatus.ACTIVE,
            using_db=tx_conn,
        )

    if conn is None:
        async with in_transaction() as tx_conn:
            await _transfer_in_db(tx_conn)
    else:
        await _transfer_in_db(conn)

    # Update Stripe customer metadata (outside transaction)
    try:
        stripe.Customer.modify(
            stripe_customer.stripe_customer_id,
            metadata={
                "organization_id": str(target_org.id),
                "organization_name": target_org.name,
                "billing_type": "team" if target_org.type == OrganizationType.TEAM else "personal",
            },
        )
        logger.info(
            "Updated Stripe customer %s metadata for org %s",
            stripe_customer.stripe_customer_id, target_org.id
        )
    except stripe.error.StripeError as exc:
        logger.error("Failed to update Stripe customer metadata: %s", exc)
        # Continue anyway - metadata update is not critical

    logger.info(
        "Transferred subscription from org %s to org %s",
        source_org.id, target_org.id
    )


async def update_org_has_payment_method(organization: Organization, has_payment_method: bool) -> None:
    """
    Update organization's has_payment_method flag.

    Called when payment method is added/removed via Stripe webhook.

    Args:
        organization: The organization to update
        has_payment_method: Whether a valid payment method is attached
    """
    organization.has_payment_method = has_payment_method
    if has_payment_method and not organization.payment_method_added_at:
        organization.payment_method_added_at = datetime.now(timezone.utc)
    await organization.save(update_fields=["has_payment_method", "payment_method_added_at"])


# ============================================================================
# User-Centric Wrapper Functions (for backward compatibility)
# ============================================================================


async def _get_user_personal_org(user: User) -> Organization:
    """Get or create the user's personal organization."""
    organization = await Organization.get_or_none(owner=user, type=OrganizationType.PERSONAL)
    if not organization:
        raise ValueError(f"No personal organization found for user {user.user_id}")
    return organization


async def get_user_subscription(user: User) -> BillingSubscription:
    """
    Get user's personal organization's billing subscription.

    Wrapper for get_org_subscription that uses user's personal org.
    """
    organization = await _get_user_personal_org(user)
    return await get_org_subscription(organization)


async def create_checkout_session(
    user: User,
    price_id: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe Checkout session for a user's personal organization.

    Wrapper for create_org_checkout_session.
    """
    organization = await _get_user_personal_org(user)
    return await create_org_checkout_session(organization, user, price_id, success_url, cancel_url)


async def create_portal_session(user: User, return_url: str) -> str:
    """
    Create a Stripe Customer Portal session for a user's personal organization.

    Wrapper for create_org_portal_session.
    """
    organization = await _get_user_personal_org(user)
    return await create_org_portal_session(organization, user, return_url)


async def get_or_create_stripe_customer(user: User) -> str:
    """
    Get or create a Stripe customer for a user's personal organization.

    Wrapper for get_or_create_org_stripe_customer.
    """
    organization = await _get_user_personal_org(user)
    return await get_or_create_org_stripe_customer(organization, user)


async def list_customer_invoices(user: User, page: int, page_size: int) -> dict:
    """
    List invoices for a user's personal organization.

    Wrapper for list_org_invoices.
    """
    organization = await _get_user_personal_org(user)
    return await list_org_invoices(organization, page, page_size)


async def list_customer_payments(user: User, page: int, page_size: int) -> dict:
    """
    List payments (charges) for a user's personal organization.

    Args:
        user: The user
        page: Page number (1-based)
        page_size: Items per page

    Returns:
        Dict with items list and has_more flag
    """
    organization = await _get_user_personal_org(user)

    if not organization.stripe_customer_id:
        return {"items": [], "has_more": False}

    stripe_customer = await StripeCustomer.get(id=organization.stripe_customer_id)

    items, has_more = _paginate_stripe_list(
        stripe.Charge.list,
        page=page,
        page_size=page_size,
        customer=stripe_customer.stripe_customer_id,
    )

    def _serialize_charge(charge: dict) -> dict[str, Any]:
        return {
            "id": charge.get("id"),
            "amount": charge.get("amount"),
            "currency": charge.get("currency"),
            "status": charge.get("status"),
            "created": charge.get("created"),
            "description": charge.get("description"),
            "receipt_url": charge.get("receipt_url"),
        }

    return {"items": [_serialize_charge(c) for c in items], "has_more": has_more}
