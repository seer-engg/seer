"""Base auth provider interface for pluggable authentication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from seer.auth.clerk_verifier import VerifiedClerkToken


class AuthProvider(ABC):
    """Abstract base for authentication providers."""

    @abstractmethod
    def verify_token(self, token: str) -> Optional[VerifiedClerkToken]:
        """Verify a token and return verified claims, or None on failure."""

    @abstractmethod
    def verify_token_with_error(self, token: str) -> tuple[Optional[VerifiedClerkToken], Optional[str]]:
        """Verify a token, returning (result, error_message)."""

    @abstractmethod
    def is_auth_required(self) -> bool:
        """Whether requests must carry a valid token."""
