"""
Custom exceptions for usage limit enforcement.

Provides structured exceptions that include metadata for upgrade prompts
and detailed error responses.
"""

from enum import Enum

from seer.database.subscription_models import SubscriptionTier


class LimitPeriod(str, Enum):
    """
    Time period for credit limits.

    Used to indicate which rolling window limit was exceeded.
    """

    FIVE_HOUR = "5_hour"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class UsageLimitError(Exception):
    """
    Base exception for all usage limit violations.

    This exception should be caught by API middleware and converted to
    HTTP 402 Payment Required responses with structured error bodies.

    Attributes:
        resource: The resource that hit the limit (e.g., "workflows", "runs")
        limit: The limit value that was exceeded
        current: The current usage value
        tier: The user's current subscription tier
        message: Human-readable error message
        upgrade_url: URL to pricing/upgrade page
    """

    def __init__(
        self,
        resource: str,
        limit: int,
        current: int,
        tier: SubscriptionTier,
        message: str,
        upgrade_url: str = "/pricing",
    ):  # pylint: disable=too-many-positional-arguments  # Exception classes need structured error data
        self.resource = resource
        self.limit = limit
        self.current = current
        self.tier = tier
        self.message = message
        self.upgrade_url = upgrade_url
        super().__init__(message)

    def to_dict(self) -> dict:
        """
        Convert exception to structured error response.

        Returns:
            Dictionary with error details suitable for API response
        """
        return {
            "error": "usage_limit_exceeded",
            "resource": self.resource,
            "limit": self.limit,
            "current": self.current,
            "tier": self.tier.value,
            "upgrade_url": self.upgrade_url,
            "message": self.message,
        }


class WorkflowLimitExceeded(UsageLimitError):
    """
    Raised when user attempts to create more workflows than allowed by their tier.
    """

    def __init__(
        self,
        limit: int,
        current: int,
        tier: SubscriptionTier,
        upgrade_url: str = "/pricing",
    ):
        message = (
            f"You've reached the maximum of {limit} workflows on the {tier.value} plan. "
            "Upgrade to Pro for unlimited workflows."
        )
        super().__init__(
            resource="workflows",
            limit=limit,
            current=current,
            tier=tier,
            message=message,
            upgrade_url=upgrade_url,
        )


class RunLimitExceeded(UsageLimitError):
    """
    Raised when user attempts to execute more workflow runs than allowed this month.
    """

    def __init__(
        self,
        limit: int,
        current: int,
        tier: SubscriptionTier,
        upgrade_url: str = "/pricing",
    ):
        message = (
            f"You've reached your monthly limit of {limit:,} workflow runs on the {tier.value} plan. "
            "Upgrade to increase your run quota."
        )
        super().__init__(
            resource="runs",
            limit=limit,
            current=current,
            tier=tier,
            message=message,
            upgrade_url=upgrade_url,
        )


class TrialExpiredError(UsageLimitError):
    """
    Raised when a Cloud Free user's 14-day trial has expired.
    """

    def __init__(
        self,
        days_since_signup: int,
        upgrade_url: str = "/pricing",
    ):
        message = (
            f"Your 14-day trial has ended ({days_since_signup} days since signup). "
            "Upgrade to Pro to continue using Seer."
        )
        super().__init__(
            resource="account_days",
            limit=14,
            current=days_since_signup,
            tier=SubscriptionTier.FREE,
            message=message,
            upgrade_url=upgrade_url,
        )


class CreditLimitExceeded(UsageLimitError):
    """
    Raised when user has exhausted their LLM credit allowance for a given period.

    Supports monthly, weekly, and 5-hour rolling window limits.
    Also supports overage pricing information for paid tier users.
    """

    # Map periods to human-readable names
    _PERIOD_NAMES = {
        LimitPeriod.FIVE_HOUR: "5-hour",
        LimitPeriod.WEEKLY: "weekly",
        LimitPeriod.MONTHLY: "monthly",
    }

    def __init__(
        self,
        limit: float,
        current: float,
        tier: SubscriptionTier,
        period: LimitPeriod = LimitPeriod.MONTHLY,
        is_soft_limit: bool = False,
        upgrade_url: str = "/pricing",
        overage_enabled: bool = False,
        overage_cap_reached: bool = False,
    ):  # pylint: disable=too-many-positional-arguments,too-many-arguments  # Exception classes need structured error data
        period_name = self._PERIOD_NAMES.get(period, period.value)
        if is_soft_limit:
            message = (
                f"Warning: You've used ${current:.2f} of your ${limit:.2f} {period_name} LLM credit allowance "
                f"on the {tier.value} plan. You're approaching your limit."
            )
        elif overage_cap_reached:
            message = (
                f"You've reached your overage spending cap. "
                f"You've used ${current:.2f} beyond your ${limit:.2f} {period_name} allowance. "
                "Increase your spending cap to continue using LLM credits."
            )
        elif overage_enabled:
            # This shouldn't normally happen if overage is enabled but not cap-reached
            message = (
                f"You've exhausted your ${limit:.2f} {period_name} LLM credit allowance on the {tier.value} plan. "
                "Check your overage settings."
            )
        else:
            message = (
                f"You've exhausted your ${limit:.2f} {period_name} LLM credit allowance on the {tier.value} plan. "
                "Upgrade to increase your LLM credits or enable usage-based pricing."
            )
        super().__init__(
            resource="llm_credits",
            limit=int(limit),  # Convert to int for consistency
            current=int(current),
            tier=tier,
            message=message,
            upgrade_url=upgrade_url,
        )
        self.period = period
        self.is_soft_limit = is_soft_limit
        self.actual_limit = limit
        self.actual_current = current
        self.overage_enabled = overage_enabled
        self.overage_cap_reached = overage_cap_reached

    def to_dict(self) -> dict:
        """Add credit-specific fields to error response."""
        data = super().to_dict()
        data["limit"] = self.actual_limit
        data["current"] = self.actual_current
        data["period"] = self.period.value
        data["is_soft_limit"] = self.is_soft_limit
        data["overage_enabled"] = self.overage_enabled
        data["overage_cap_reached"] = self.overage_cap_reached
        return data


class RunCostCapExceeded(Exception):
    """Raised when per-execution cost cap is exceeded."""

    def __init__(
        self,
        run_identifier: str,
        accumulated_cost: float,
        cost_cap: float,
        run_type: str,  # "workflow" or "chat"
    ):
        self.run_identifier = run_identifier
        self.accumulated_cost = accumulated_cost
        self.cost_cap = cost_cap
        self.run_type = run_type

        message = (
            f"{run_type.capitalize()} run '{run_identifier}' exceeded cost cap: "
            f"${accumulated_cost:.2f} > ${cost_cap:.2f}"
        )
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "error": "run_cost_cap_exceeded",
            "run_identifier": self.run_identifier,
            "accumulated_cost": round(self.accumulated_cost, 2),
            "cost_cap": round(self.cost_cap, 2),
            "run_type": self.run_type,
            "message": str(self),
        }


class PollingIntervalTooFast(UsageLimitError):
    """
    Raised when user attempts to set a polling interval faster than their tier allows.

    This is typically a soft error - the system will clamp to the minimum allowed value
    and warn the user.
    """

    def __init__(
        self,
        requested_interval: int,
        min_interval: int,
        tier: SubscriptionTier,
        upgrade_url: str = "/pricing",
    ):
        message = (
            f"The {tier.value} plan allows polling intervals of {min_interval}s or slower. "
            f"Your requested {requested_interval}s has been adjusted to {min_interval}s. "
            "Upgrade to Pro+ for faster polling."
        )
        super().__init__(
            resource="polling_interval",
            limit=min_interval,
            current=requested_interval,
            tier=tier,
            message=message,
            upgrade_url=upgrade_url,
        )
        self.requested_interval = requested_interval
        self.min_interval = min_interval

    def to_dict(self) -> dict:
        """Add polling-specific fields to error response."""
        data = super().to_dict()
        data["requested_interval"] = self.requested_interval
        data["min_interval"] = self.min_interval
        data["clamped_to"] = self.min_interval
        return data
