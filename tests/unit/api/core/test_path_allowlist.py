"""Unit tests for path allowlist helpers."""
import pytest

from seer.api.core.middleware.path_allowlist import (
    is_payment_exempt_path,
    is_public_path,
)


class TestIsPublicPath:
    """Test is_public_path() function."""

    def test_public_paths(self):
        """Test exact public path matches."""
        assert is_public_path("/health")
        assert is_public_path("/api/subscriptions/webhooks/stripe")
        assert is_public_path("/sentry-debug")

    def test_public_path_normalization(self):
        """Test trailing slash normalization."""
        assert is_public_path("/health/")
        assert is_public_path("/api/subscriptions/webhooks/stripe/")

    def test_public_prefixes(self):
        """Test public prefix matches."""
        assert is_public_path("/api/v1/webhooks/some-webhook")
        assert is_public_path("/api/forms/submit")
        assert is_public_path("/sse/events")
        assert is_public_path("/mcp/transport")

    def test_oauth_callbacks(self):
        """Test OAuth callback pattern matching."""
        assert is_public_path("/api/integrations/google/callback")
        assert is_public_path("/api/integrations/slack/callback")
        assert is_public_path("/api/integrations/custom/callback")

    def test_docs_paths_when_included(self):
        """Test docs paths are public when include_docs=True."""
        assert is_public_path("/docs", include_docs=True)
        assert is_public_path("/openapi.json", include_docs=True)

    def test_docs_paths_when_not_included(self):
        """Test docs paths are not public when include_docs=False."""
        assert not is_public_path("/docs", include_docs=False)
        assert not is_public_path("/openapi.json", include_docs=False)

    def test_invitation_token_paths(self):
        """Test invitation paths: token details public, accept/decline require auth."""
        # GET invitation details (view before signing in) - public
        assert is_public_path("/api/organizations/invitations/abc123")
        assert is_public_path("/api/organizations/invitations/some-uuid-token")

        # POST accept/decline - NOT public (require auth)
        assert not is_public_path("/api/organizations/invitations/abc123/accept")
        assert not is_public_path("/api/organizations/invitations/abc123/decline")
        assert not is_public_path("/api/organizations/invitations/some-uuid-token/accept")
        assert not is_public_path("/api/organizations/invitations/some-uuid-token/decline")

        # Base path without token - NOT public
        assert not is_public_path("/api/organizations/invitations")
        assert not is_public_path("/api/organizations/invitations/")

    def test_non_public_paths(self):
        """Test non-public paths return False."""
        assert not is_public_path("/api/v1/workflows")
        assert not is_public_path("/api/subscriptions/current")
        assert not is_public_path("/api/usage")


class TestIsPaymentExemptPath:
    """Test is_payment_exempt_path() function."""

    def test_exact_payment_exempt_paths(self):
        """Test exact payment-exempt path matches."""
        assert is_payment_exempt_path("/api/subscriptions/current")
        assert is_payment_exempt_path("/api/subscriptions/checkout")
        assert is_payment_exempt_path("/api/subscriptions/portal")
        assert is_payment_exempt_path("/api/subscriptions/invoices")
        assert is_payment_exempt_path("/api/subscriptions/payments")
        assert is_payment_exempt_path("/api/subscriptions/create-with-trial")
        assert is_payment_exempt_path("/api/subscriptions/setup-intent")
        assert is_payment_exempt_path("/api/subscriptions/setup-intent/confirm")
        assert is_payment_exempt_path("/api/subscriptions/payment-method/status")
        assert is_payment_exempt_path("/api/usage")
        assert is_payment_exempt_path("/api/users/me/settings")

    def test_payment_exempt_path_normalization(self):
        """Test trailing slash normalization."""
        assert is_payment_exempt_path("/api/subscriptions/current/")
        assert is_payment_exempt_path("/api/usage/")
        assert is_payment_exempt_path("/api/users/me/settings/")

    def test_payment_exempt_prefixes(self):
        """Test payment-exempt prefix matches."""
        assert is_payment_exempt_path("/api/usage/analytics")
        assert is_payment_exempt_path("/api/usage/analytics/daily")
        assert is_payment_exempt_path("/api/usage/analytics/monthly/workflows")

    def test_non_exempt_paths(self):
        """Test non-exempt paths return False."""
        assert not is_payment_exempt_path("/api/v1/workflows")
        assert not is_payment_exempt_path("/api/v1/workflows/wf_123/run")
        assert not is_payment_exempt_path("/api/nexus/chat")
        assert not is_payment_exempt_path("/health")

    def test_extra_allowed_paths(self):
        """Test extra_allowed_paths parameter."""
        assert not is_payment_exempt_path("/api/custom/endpoint")
        assert is_payment_exempt_path(
            "/api/custom/endpoint",
            extra_allowed_paths=["/api/custom/endpoint"]
        )

    def test_public_vs_payment_exempt(self):
        """Test that public paths and payment-exempt paths are distinct."""
        # Public paths should not be payment-exempt
        assert not is_payment_exempt_path("/health")
        assert not is_payment_exempt_path("/api/subscriptions/webhooks/stripe")

        # Payment-exempt paths should not be public
        assert not is_public_path("/api/subscriptions/current")
        assert not is_public_path("/api/usage")
