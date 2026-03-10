"""Database models for subscription management."""
from enum import Enum

from tortoise import fields, models


class StripeCustomer(models.Model):
    """
    Tracks Stripe customer ownership with full audit trail.

    This table serves as the source of truth for Stripe customer assignments.
    Benefits:
    - Preserves who originally created the customer (audit trail)
    - When subscription transfers, customer stays same, just reassigned to new org
    - Webhook lookups: StripeCustomer.get(stripe_customer_id=...) → organization
    """

    id = fields.IntField(primary_key=True)
    stripe_customer_id = fields.CharField(max_length=255, unique=True)

    # Who originally created this Stripe customer (audit trail - never changes)
    created_by_user = fields.ForeignKeyField(
        "models.User",
        related_name="created_stripe_customers",
        on_delete=fields.CASCADE,
    )

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "stripe_customers"

    def __str__(self) -> str:
        # pylint: disable=no-member  # Reason: Tortoise ORM generates FK _id attributes dynamically
        return f"StripeCustomer<id={self.id}, stripe_id={self.stripe_customer_id}, created_by={self.created_by_user_id}>"


class SubscriptionTier(str, Enum):
    """Available subscription tiers."""
    FREE = "free"
    PRO = "pro"
    PRO_PLUS = "pro_plus"


class SubscriptionStatus(str, Enum):
    """Subscription status values mapped from Stripe."""
    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"


class StripeWebhookEventStatus(str, Enum):
    """State machine for webhook processing."""
    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class BillingSubscription(models.Model):
    """Subscription record tied to an organization.

    Every organization (personal or team) has exactly one subscription.
    Personal orgs start with FREE tier, can upgrade via checkout.
    Team orgs either inherit from transfer or start FREE.
    """

    id = fields.IntField(primary_key=True)

    # Organization-centric billing - every org has exactly one subscription
    organization = fields.OneToOneField(
        "models.Organization",
        related_name="billing_subscription",
        on_delete=fields.CASCADE,
    )

    stripe_subscription_id = fields.CharField(max_length=255, unique=True, null=True)

    tier = fields.CharEnumField(SubscriptionTier, default=SubscriptionTier.FREE)
    status = fields.CharEnumField(SubscriptionStatus, default=SubscriptionStatus.ACTIVE)

    current_period_start = fields.DatetimeField(null=True)
    current_period_end = fields.DatetimeField(null=True)
    cancel_at_period_end = fields.BooleanField(default=False)
    is_early_adopter = fields.BooleanField(default=False)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "billing_subscriptions"

    def __str__(self) -> str:
        return f"BillingSubscription<profile={self.id}, tier={self.tier.value}>"


class EarlyAdopterCounter(models.Model):
    """Counter for early adopter slots per tier."""

    id = fields.IntField(primary_key=True)
    tier = fields.CharField(max_length=20, unique=True)
    count = fields.IntField(default=0)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "early_adopter_counters"

    def __str__(self) -> str:
        return f"EarlyAdopterCounter<tier={self.tier}, count={self.count}>"


class StripeWebhookEvent(models.Model):
    """
    Durable storage for Stripe webhook events to enable idempotent processing.
    """

    id = fields.IntField(primary_key=True)
    event_id = fields.CharField(max_length=255, unique=True)
    type = fields.CharField(max_length=255)
    payload = fields.JSONField()
    status = fields.CharEnumField(
        StripeWebhookEventStatus,
        default=StripeWebhookEventStatus.RECEIVED,
    )
    attempts = fields.IntField(default=0)
    last_error = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "stripe_webhook_events"

    def __str__(self) -> str:
        return f"StripeWebhookEvent<event_id={self.event_id}, status={self.status}>"
