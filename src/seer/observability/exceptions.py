"""
Custom exceptions for usage limit enforcement.

Provides structured exceptions that include metadata for upgrade prompts
and detailed error responses.
"""
from typing import Optional

from seer.database.subscription_models import SubscriptionTier


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
    ):
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


class MessageLimitExceeded(UsageLimitError):
    """
    Raised when user attempts to send more chat messages than allowed GLOBALLY.

    Changed from per-workflow to global (across all workflows) limit.
    """

    def __init__(
        self,
        limit: int,
        current: int,
        tier: SubscriptionTier,
        upgrade_url: str = "/pricing",
    ):
        message = (
            f"You've reached your limit of {limit} total chat messages on the {tier.value} plan. "
            "Upgrade to Pro for unlimited chat messages."
        )
        super().__init__(
            resource="chat_messages",
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
    Raised when user has exhausted their monthly LLM credit allowance.
    """

    def __init__(
        self,
        limit: float,
        current: float,
        tier: SubscriptionTier,
        is_soft_limit: bool = False,
        upgrade_url: str = "/pricing",
    ):
        if is_soft_limit:
            message = (
                f"Warning: You've used ${current:.2f} of your ${limit:.2f} monthly LLM credit allowance "
                f"on the {tier.value} plan. You're approaching your limit."
            )
        else:
            message = (
                f"You've exhausted your ${limit:.2f} monthly LLM credit allowance on the {tier.value} plan. "
                "Upgrade to increase your LLM credits."
            )
        super().__init__(
            resource="llm_credits",
            limit=int(limit),  # Convert to int for consistency
            current=int(current),
            tier=tier,
            message=message,
            upgrade_url=upgrade_url,
        )
        self.is_soft_limit = is_soft_limit
        self.actual_limit = limit
        self.actual_current = current

    def to_dict(self) -> dict:
        """Add credit-specific fields to error response."""
        data = super().to_dict()
        data["limit"] = self.actual_limit
        data["current"] = self.actual_current
        data["is_soft_limit"] = self.is_soft_limit
        return data


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


class ChatDisabledError(Exception):
    """
    Raised when chat AI is accessed in self-hosted mode where it's disabled.

    Returns HTTP 403 Forbidden (not 402) since this isn't an upgradeable limitation.
    """

    def __init__(self):
        message = (
            "Chat AI is not available in self-hosted mode. "
            "Use your own LLM API keys (BYOK) for AI functionality."
        )
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert to structured error response."""
        return {
            "error": "feature_disabled",
            "resource": "chat_ai",
            "message": str(self),
        }
