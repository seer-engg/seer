"""
Database models for usage tracking and enforcement.

These models track resource consumption across different dimensions
(workflows, runs, LLM credits) to enforce subscription limits.
"""
from enum import Enum

from tortoise import fields, models


class ResourceType(str, Enum):
    """Types of resources that can be tracked."""

    WORKFLOWS = "workflows"  # Total workflow count
    RUNS = "runs"  # Workflow execution runs
    LLM_CREDITS = "llm_credits"  # LLM usage cost in USD


class UsageCounter(models.Model):
    """
    Tracks usage counts for various resources per user/organization and time period.

    Supports both total counts (all-time) and windowed counts (monthly).
    Designed for efficient querying and Redis caching.

    Usage patterns:
    - User-level: user is set, organization is null (personal org usage)
    - Org-level: organization is set, user is null (team org aggregate usage)

    Example queries:
    - Total workflows for user: resource_type=WORKFLOWS, period_start=None, user=user
    - Org runs this month: resource_type=RUNS, period_start=2024-01-01, organization=org
    """

    id = fields.IntField(primary_key=True)

    # User-level tracking (for personal orgs or per-member breakdown)
    user = fields.ForeignKeyField(
        "models.User",
        related_name="usage_counters",
        on_delete=fields.CASCADE,
        db_index=True,
        null=True,
    )

    # Organization-level tracking (for team orgs)
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="usage_counters",
        on_delete=fields.CASCADE,
        db_index=True,
        null=True,
    )

    resource_type = fields.CharEnumField(ResourceType, db_index=True)

    # Time window for the counter (None = all-time)
    period_start = fields.DatetimeField(null=True, db_index=True)
    period_end = fields.DatetimeField(null=True)

    # The actual count (incremented atomically)
    count = fields.BigIntField(default=0)

    # Monetary value (for LLM_CREDITS resource type, in USD)
    value = fields.DecimalField(max_digits=10, decimal_places=2, default=0.0)

    # Optional reference to specific resource (e.g., workflow_id)
    reference_id = fields.CharField(max_length=255, null=True, db_index=True)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "usage_counters"
        indexes = [
            # Primary lookup: user + resource + period
            ("user_id", "resource_type", "period_start"),
            # Lookup by resource reference
            ("user_id", "resource_type", "reference_id"),
            # Org-level lookup: org + resource + period
            ("organization_id", "resource_type", "period_start"),
        ]

    def __str__(self) -> str:
        period_str = f"{self.period_start}" if self.period_start else "all-time"
        owner = f"user={self.user_id}" if self.user_id else f"org={self.organization_id}"
        return f"UsageCounter<{owner}, type={self.resource_type.value}, period={period_str}, count={self.count}>"


class LLMUsageRecord(models.Model):
    """
    Detailed log of individual LLM API calls for cost tracking.

    Aggregated into UsageCounter for limit checking, but preserved
    for detailed analytics and debugging.

    Organization field enables team-level cost tracking and billing.
    """

    id = fields.IntField(primary_key=True)

    # User who triggered the LLM call
    user = fields.ForeignKeyField(
        "models.User",
        related_name="llm_usage_records",
        on_delete=fields.CASCADE,
        db_index=True,
        null=True,
    )

    # Organization context (for team billing aggregation)
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="llm_usage_records",
        on_delete=fields.CASCADE,
        db_index=True,
        null=True,
    )

    # Optional reference to workflow run
    workflow_run_id = fields.CharField(max_length=255, null=True, db_index=True)

    # LLM provider and model
    provider = fields.CharField(max_length=50)  # e.g., "openai", "anthropic"
    model = fields.CharField(max_length=100, db_index=True)  # e.g., "gpt-4", "claude-3-opus"

    # Token usage
    input_tokens = fields.IntField(default=0)
    output_tokens = fields.IntField(default=0)
    total_tokens = fields.IntField(default=0)

    # Cost in USD (calculated using model pricing)
    cost = fields.DecimalField(max_digits=10, decimal_places=6)

    # Additional context
    operation = fields.CharField(max_length=100, null=True)  # e.g., "workflow_execution", "chat_message"
    metadata = fields.JSONField(null=True)  # Flexible storage for additional data

    created_at = fields.DatetimeField(auto_now_add=True, db_index=True)

    class Meta:
        table = "llm_usage_records"
        indexes = [
            # Monthly cost queries per user
            ("user_id", "created_at"),
            # Workflow run analysis
            ("workflow_run_id", "created_at"),
            # Model usage analytics
            ("model", "created_at"),
            # Organization-level cost queries
            ("organization_id", "created_at"),
        ]

    def __str__(self) -> str:
        owner = f"user={self.user_id}" if self.user_id else f"org={self.organization_id}"
        return f"LLMUsageRecord<{owner}, model={self.model}, tokens={self.total_tokens}, cost=${self.cost}>"
