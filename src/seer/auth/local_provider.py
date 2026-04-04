"""Local auth provider for self-hosted / open-source deployments.

When AUTH_PROVIDER=local, no external auth service is needed.
All requests are authenticated as a default local user.
An optional X-User-ID header can override the user identity.
"""

from __future__ import annotations

from typing import Optional

from seer.auth.base import AuthProvider
from seer.auth.clerk_verifier import VerifiedClerkToken

LOCAL_DEFAULT_USER_ID = "local-default-user"
LOCAL_DEFAULT_EMAIL = "local@seer.local"


class LocalAuthProvider(AuthProvider):
    """No-op auth provider that always authenticates as a local user."""

    def verify_token(self, token: str) -> Optional[VerifiedClerkToken]:
        return self._make_local_token()

    def verify_token_with_error(self, token: str) -> tuple[Optional[VerifiedClerkToken], Optional[str]]:
        return self._make_local_token(), None

    def is_auth_required(self) -> bool:
        return False

    def authenticate_local(self, user_id: Optional[str] = None) -> VerifiedClerkToken:
        """Create a local user token, optionally with a custom user_id from X-User-ID header."""
        return self._make_local_token(user_id=user_id)

    def _make_local_token(self, user_id: Optional[str] = None) -> VerifiedClerkToken:
        return VerifiedClerkToken(
            user_id=user_id or LOCAL_DEFAULT_USER_ID,
            email=LOCAL_DEFAULT_EMAIL,
            first_name="Local",
            last_name="User",
            claims={},
        )
