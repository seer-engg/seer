"""
Database models for usage-based pricing (overages) for LLM credits.

These models track overage settings per billing profile and individual
overage usage records for Stripe reporting.
"""
from decimal import Decimal
from enum import Enum

from tortoise import fields, models


class OverageRecordStatus(str, Enum):
    """Status of an overage usage record."""

    PENDING = "pending"  # Created, awaiting Stripe reporting
    REPORTED = "reported"  # Successfully reported to Stripe
    FAILED = "failed"  # Failed to report to Stripe


class OverageSettings(models.Model):
    """
    Overage pricing settings for a billing profile.

    Allows paid tier users to opt into usage-based pricing for LLM credits
    beyond their subscription allowance.
    """

    id = fields.IntField(primary_key=True)

    billing_profile = fields.OneToOneField(
        "models.BillingProfile",
        related_name="overage_settings",
        on_delete=fields.CASCADE,
    )

    # Whether usage-based pricing is enabled
    enabled = fields.BooleanField(default=False)

    # Spending cap in cents (default $50)
    spending_cap_cents = fields.IntField(default=5000)

    # Margin multiplier applied to LLM cost (default 30% margin = 1.30x)
    margin_multiplier = fields.DecimalField(max_digits=5, decimal_places=2, default=Decimal("1.30"))

    # Current period overage tracking (in cents)
    current_period_overage_cents = fields.IntField(default=0)

    # Start of current billing period (aligned with subscription)
    current_period_start = fields.DatetimeField(null=True)

    # Stripe metered subscription item ID for usage reporting
    stripe_metered_subscription_item_id = fields.CharField(max_length=255, null=True)

    # When overage was enabled
    enabled_at = fields.DatetimeField(null=True)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "overage_settings"

    def __str__(self) -> str:
        # pylint: disable=no-member  # billing_profile_id is dynamically created by Tortoise ORM
        return f"OverageSettings<profile={self.billing_profile_id}, enabled={self.enabled}, cap=${self.spending_cap_cents / 100:.2f}>"

    @property
    def spending_cap_dollars(self) -> Decimal:
        """Return spending cap in dollars."""
        return Decimal(self.spending_cap_cents) / Decimal(100)

    @property
    def current_period_overage_dollars(self) -> Decimal:
        """Return current period overage in dollars."""
        return Decimal(self.current_period_overage_cents) / Decimal(100)

    @property
    def remaining_cap_cents(self) -> int:
        """Return remaining spending cap in cents."""
        return max(0, self.spending_cap_cents - self.current_period_overage_cents)

    def is_cap_reached(self) -> bool:
        """Check if spending cap has been reached."""
        return self.current_period_overage_cents >= self.spending_cap_cents


class OverageUsageRecord(models.Model):
    """
    Individual overage usage record for Stripe reporting.

    Tracks each overage charge with its base cost, billed amount (with margin),
    and Stripe reporting status.
    """

    id = fields.IntField(primary_key=True)

    overage_settings = fields.ForeignKeyField(
        "models.OverageSettings",
        related_name="usage_records",
        on_delete=fields.CASCADE,
    )

    # Optional reference to the LLM usage record that triggered this overage
    llm_usage_record = fields.ForeignKeyField(
        "models.LLMUsageRecord",
        related_name="overage_records",
        on_delete=fields.SET_NULL,
        null=True,
    )

    # Base cost in cents (actual LLM cost)
    base_cost_cents = fields.IntField()

    # Billed amount in cents (cost × margin)
    billed_amount_cents = fields.IntField()

    # Stripe usage record ID for tracking
    stripe_usage_record_id = fields.CharField(max_length=255, null=True)

    # When the usage was reported to Stripe
    reported_to_stripe_at = fields.DatetimeField(null=True)

    # Status of Stripe reporting
    status = fields.CharEnumField(OverageRecordStatus, default=OverageRecordStatus.PENDING)

    # Error message if reporting failed
    error_message = fields.TextField(null=True)

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "overage_usage_records"
        indexes = [
            ("overage_settings_id", "status"),
            ("overage_settings_id", "created_at"),
        ]

    def __str__(self) -> str:
        # pylint: disable=no-member  # overage_settings_id is dynamically created by Tortoise ORM
        base = self.base_cost_cents / 100
        billed = self.billed_amount_cents / 100
        return f"OverageUsageRecord<settings={self.overage_settings_id}, base=${base:.4f}, billed=${billed:.4f}, status={self.status}>"
