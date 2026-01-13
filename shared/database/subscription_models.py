"""
Database models for subscription management.

Tracks user subscription state synced from Stripe webhooks.
"""
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


class UserSubscription(models.Model):
    """
    Tracks user subscription state from Stripe.

    One-to-one relationship with User. Created on first subscription
    lookup with FREE tier as default.
    """

    id = fields.IntField(primary_key=True)
    user = fields.OneToOneField(
        "models.User",
        related_name="subscription",
        on_delete=fields.CASCADE,
    )

    # Stripe identifiers
    stripe_customer_id = fields.CharField(max_length=255, unique=True, null=True)
    stripe_subscription_id = fields.CharField(max_length=255, unique=True, null=True)

    # Subscription state
    tier = fields.CharEnumField(SubscriptionTier, default=SubscriptionTier.FREE)
    status = fields.CharEnumField(SubscriptionStatus, default=SubscriptionStatus.ACTIVE)

    # Billing period info
    current_period_start = fields.DatetimeField(null=True)
    current_period_end = fields.DatetimeField(null=True)
    cancel_at_period_end = fields.BooleanField(default=False)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_subscriptions"

    def __str__(self) -> str:
        return f"UserSubscription<user={self.user_id}, tier={self.tier}>"


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
