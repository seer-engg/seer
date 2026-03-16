"""
Computed properties mixin for SeerConfig.

This module is separated from config.py to reduce line count while maintaining
all property-based logic in a clean, testable location.
"""
import hashlib


class SeerConfigPropertiesMixin:
    """Mixin providing computed properties for SeerConfig.

    Attributes are accessed via self and resolved at runtime when mixed into SeerConfig.
    """

    @property
    def is_stripe_configured(self) -> bool:
        """Check if Stripe is configured for subscription billing."""
        return self.stripe_secret_key is not None and self.stripe_webhook_secret is not None  # type: ignore[attr-defined]

    @property
    def is_slack_configured(self) -> bool:
        """Check if Slack error notifications are configured."""
        return self.slack_bot_token is not None and self.slack_error_channel_id is not None  # type: ignore[attr-defined]

    @property
    def is_langfuse_configured(self) -> bool:
        """Check if Langfuse is configured (at least one project has credentials)."""
        if not self.langfuse_enabled:  # type: ignore[attr-defined]
            return False
        nexus_configured = (
            self.langfuse_nexus_public_key is not None and self.langfuse_nexus_secret_key is not None  # type: ignore[attr-defined]
        )
        workflow_configured = (
            self.langfuse_workflow_public_key is not None and self.langfuse_workflow_secret_key is not None  # type: ignore[attr-defined]
        )
        return nexus_configured or workflow_configured

    @property
    def is_langfuse_nexus_configured(self) -> bool:
        """Check if Langfuse is configured for Nexus agent tracing."""
        return (
            self.langfuse_enabled  # type: ignore[attr-defined]
            and self.langfuse_nexus_public_key is not None  # type: ignore[attr-defined]
            and self.langfuse_nexus_secret_key is not None  # type: ignore[attr-defined]
        )

    @property
    def is_langfuse_workflow_configured(self) -> bool:
        """Check if Langfuse is configured for Workflow tracing."""
        return (
            self.langfuse_enabled  # type: ignore[attr-defined]
            and self.langfuse_workflow_public_key is not None  # type: ignore[attr-defined]
            and self.langfuse_workflow_secret_key is not None  # type: ignore[attr-defined]
        )

    @property
    def is_posthog_configured(self) -> bool:
        """Check if PostHog analytics is configured and enabled."""
        return self.posthog_enabled and self.posthog_api_key is not None  # type: ignore[attr-defined]

    @property
    def is_sentry_configured(self) -> bool:
        """Check if Sentry error monitoring is configured and enabled."""
        return self.sentry_enabled and self.sentry_dsn is not None  # type: ignore[attr-defined]

    @property
    def browser_encryption_key_bytes(self) -> bytes:
        """Get 32-byte Fernet key for browser session encryption.

        Uses BROWSER_SESSION_ENCRYPTION_KEY env var if set,
        otherwise derives from SECRET_KEY via SHA-256.
        """
        import base64  # pylint: disable=import-outside-toplevel  # Reason: only needed for this property
        import os  # pylint: disable=import-outside-toplevel  # Reason: only needed for fallback

        if self.browser_session_encryption_key:  # type: ignore[attr-defined]
            key = self.browser_session_encryption_key.encode()  # type: ignore[attr-defined]
            if len(key) == 44:
                return key
            raw = hashlib.sha256(key).digest()
            return base64.urlsafe_b64encode(raw)
        secret = os.getenv("SECRET_KEY", "dev_secret_key")
        raw = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(raw)

    @property
    def is_workflow_file_system_configured(self) -> bool:
        """Check if the workflow file system (S3/R2) is configured."""
        return self.workflow_file_s3_bucket is not None  # type: ignore[attr-defined]
