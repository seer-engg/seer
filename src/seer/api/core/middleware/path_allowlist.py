"""Shared path allowlist helpers for middleware."""
from __future__ import annotations

from typing import Iterable, Optional, Set

# Paths that should always be accessible without authentication/usage checks.
DEFAULT_PUBLIC_PATHS = {
    "/health",
    "/api/subscriptions/webhooks/stripe",
    "/api/integrations/google/callback",
    "/api/integrations/github/callback",
    "/api/integrations/supabase_mgmt/callback",
}

DEFAULT_DOCS_PATHS = {
    "/docs",
    "/openapi.json",
}

# Prefixes that represent collections of public endpoints.
DEFAULT_PUBLIC_PREFIXES = (
    "/api/v1/webhooks",
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

    for prefix in DEFAULT_PUBLIC_PREFIXES:
        normalized_prefix = _normalize_path(prefix)
        if normalized_path == normalized_prefix or normalized_path.startswith(f"{normalized_prefix}/"):
            return True

    return False
