"""Shared path allowlist helpers for middleware."""
from __future__ import annotations

from typing import Iterable, Optional, Set

# Paths that should always be accessible without authentication/usage checks.
DEFAULT_PUBLIC_PATHS = {
    "/health",
    "/api/auth/config",
    "/api/subscriptions/webhooks/stripe",
    "/api/integrations/google/callback",
    "/api/integrations/github/callback",
    "/api/integrations/supabase_mgmt/callback",
    "/.well-known/oauth-protected-resource",
    "/sentry-debug",  # For testing Sentry integration; not included in DEFAULT_DOCS_PATHS since it's not a documented API endpoint.
}

DEFAULT_DOCS_PATHS = {
    "/docs",
    "/openapi.json",
}

# Prefixes that represent collections of public endpoints.
DEFAULT_PUBLIC_PREFIXES = (
    "/api/v1/webhooks",
    "/api/forms",
    "/sse",  # MCP SSE transport (has its own auth)
    "/mcp",  # MCP HTTP transport (has its own auth)
    "/api/browser/recordings/shared",  # Public replay links
    "/api/public",
    "/v1/track",  # Email tracking pixel/click endpoints (recipients aren't Seer users)
)

# Payment-exempt paths: require auth but skip payment gates and usage limits.
# These are typically payment/billing endpoints that users need to access
# to resolve payment issues or add payment methods.
DEFAULT_PAYMENT_EXEMPT_PATHS = {
    "/api/subscriptions/pricing",
    "/api/subscriptions/current",
    "/api/subscriptions/checkout",
    "/api/subscriptions/portal",
    "/api/subscriptions/invoices",
    "/api/subscriptions/payments",
    "/api/subscriptions/create-with-trial",
    "/api/subscriptions/setup-intent",
    "/api/subscriptions/setup-intent/confirm",
    "/api/subscriptions/payment-method/status",
    "/api/usage",
    "/api/users/me/settings",
}

DEFAULT_PAYMENT_EXEMPT_PREFIXES = (
    "/api/usage/analytics",  # All analytics endpoints
)


def _normalize_path(path: str) -> str:
    if not path or path == "/":
        return "/"
    return path.rstrip("/")


def is_public_path(
    path: str,
    extra_allowed_paths: Optional[Iterable[str]] = None,
    *,
    include_docs: bool = False,
) -> bool:
    """
    Returns True if the request path should skip auth/usage enforcement.

    Includes a shared default allowlist plus any caller-supplied paths.
    """
    normalized_path = _normalize_path(path)

    allowed_paths: Set[str] = {_normalize_path(p) for p in DEFAULT_PUBLIC_PATHS}
    if include_docs:
        allowed_paths.update(_normalize_path(p) for p in DEFAULT_DOCS_PATHS)
    if extra_allowed_paths:
        allowed_paths.update(_normalize_path(p) for p in extra_allowed_paths)

    if normalized_path in allowed_paths:
        return True

    # OAuth-style callbacks (provider agnostic)
    if "/integrations/" in normalized_path and normalized_path.endswith("/callback"):
        return True

    # Invitation details pages (view invitation before signing in)
    # Match: /api/organizations/invitations/{token} (GET details)
    # Don't match: /api/organizations/invitations/{token}/accept
    # Don't match: /api/organizations/invitations/{token}/decline
    if normalized_path.startswith("/api/organizations/invitations/"):
        remainder = normalized_path[len("/api/organizations/invitations/"):]
        if remainder and "/" not in remainder:
            return True

    for prefix in DEFAULT_PUBLIC_PREFIXES:
        normalized_prefix = _normalize_path(prefix)
        if normalized_path == normalized_prefix or normalized_path.startswith(f"{normalized_prefix}/"):
            return True

    return False


def is_payment_exempt_path(
    path: str,
    extra_allowed_paths: Optional[Iterable[str]] = None,
) -> bool:
    """
    Returns True if the request path should skip payment gates and usage limits
    but still require authentication.

    These are typically payment/billing endpoints that users need to access
    to resolve payment issues or add payment methods.
    """
    normalized_path = _normalize_path(path)

    allowed_paths: Set[str] = {_normalize_path(p) for p in DEFAULT_PAYMENT_EXEMPT_PATHS}
    if extra_allowed_paths:
        allowed_paths.update(_normalize_path(p) for p in extra_allowed_paths)

    if normalized_path in allowed_paths:
        return True

    for prefix in DEFAULT_PAYMENT_EXEMPT_PREFIXES:
        normalized_prefix = _normalize_path(prefix)
        if normalized_path == normalized_prefix or normalized_path.startswith(f"{normalized_prefix}/"):
            return True

    return False
