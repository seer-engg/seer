"""Database models for subscription management."""
from enum import Enum

from tortoise import fields, models


class SubscriptionTier(str, Enum):
    """Available subscription tiers."""
    FREE = "free"
    PRO = "pro"
    PRO_PLUS = "pro_plus"
    ULTRA = "ultra"


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


class BillingProfileType(str, Enum):
    """Types of billing profiles."""
    INDIVIDUAL = "individual"
    TEAM = "team"


class BillingProfile(models.Model):
    """Billing profile for a paying entity (individual user or team)."""

    id = fields.IntField(primary_key=True)
    type = fields.CharEnumField(BillingProfileType, default=BillingProfileType.INDIVIDUAL)
    owner_user = fields.OneToOneField(
        "models.User",
        related_name="billing_profiles",
        on_delete=fields.CASCADE,
    )
    stripe_customer_id = fields.CharField(max_length=255, unique=True, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "billing_profiles"

    def __str__(self) -> str:
        return f"BillingProfile<id={self.id}, type={self.type.value}>"


class BillingSubscription(models.Model):
    """Subscription record tied to a billing profile."""

    id = fields.IntField(primary_key=True)
    billing_profile = fields.OneToOneField(
        "models.BillingProfile",
        related_name="subscription",
        on_delete=fields.CASCADE,
    )

    stripe_subscription_id = fields.CharField(max_length=255, unique=True, null=True)

    tier = fields.CharEnumField(SubscriptionTier, default=SubscriptionTier.FREE)
    status = fields.CharEnumField(SubscriptionStatus, default=SubscriptionStatus.ACTIVE)

    current_period_start = fields.DatetimeField(null=True)
    current_period_end = fields.DatetimeField(null=True)
    cancel_at_period_end = fields.BooleanField(default=False)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "billing_subscriptions"

    def __str__(self) -> str:
        return f"BillingSubscription<profile={self.id}, tier={self.tier.value}>"


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
