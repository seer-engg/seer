"""
Shared Clerk JWT verification logic.

This module provides JWT verification that can be used by both:
- FastAPI middleware (for the main API)
- MCP server middleware (for ChatGPT integration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import httpx
import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError

from seer.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VerifiedClerkToken:
    """Result of successful Clerk JWT verification."""

    user_id: str
    email: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    claims: Dict[str, Any]

    @property
    def scopes(self) -> List[str]:
        """Extract scopes from claims if present."""
        scope = self.claims.get("scope", "")
        if isinstance(scope, str):
            return scope.split() if scope else []
        return list(scope) if scope else []


class ClerkJWTVerifier:
    """
    Verifies Clerk-issued JWT tokens using JWKS.

    Implements the AuthProvider interface for use with the pluggable auth system.

    Example:
        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.your-domain.com/.well-known/jwks.json",
            issuer="https://clerk.your-domain.com",
            audience=["api.your-domain.com"],
        )
        result = verifier.verify_token(token)
        if result:
            print(f"Authenticated user: {result.user_id}")
    """

    def __init__(
        self,
        jwks_url: str,
        issuer: str,
        audience: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the JWT verifier.

        Args:
            jwks_url: URL to Clerk's JWKS endpoint (e.g., https://clerk.your-domain.com/.well-known/jwks.json)
            issuer: Expected JWT issuer (e.g., https://clerk.your-domain.com)
            audience: Optional list of expected audience values
        """
        if not jwks_url:
            raise ValueError("jwks_url is required for ClerkJWTVerifier")
        if not issuer:
            raise ValueError("issuer is required for ClerkJWTVerifier")

        self._jwks_client = PyJWKClient(jwks_url)
        self._issuer = issuer
        self._audience = list(audience) if audience else None

    def is_auth_required(self) -> bool:
        return True

    def verify_token(self, token: str) -> Optional[VerifiedClerkToken]:
        """
        Verify a JWT token and return the verified claims.

        Args:
            token: The JWT token string (without "Bearer " prefix)

        Returns:
            VerifiedClerkToken if verification succeeds, None if it fails
        """
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
                options={"verify_aud": self._audience is not None},
            )
            return self._create_verified_token(claims)
        except InvalidTokenError as exc:
            logger.debug("JWT validation failed: %s", exc)
            return None
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for unknown JWT validation errors
            logger.warning("Unexpected JWT verification error: %s", exc)
            return None

    def verify_token_with_error(self, token: str) -> tuple[Optional[VerifiedClerkToken], Optional[str]]:
        """
        Verify a JWT token and return both result and error message.

        This is useful when you need to return specific error messages to clients.

        Args:
            token: The JWT token string (without "Bearer " prefix)

        Returns:
            Tuple of (VerifiedClerkToken or None, error_message or None)
        """
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
                options={"verify_aud": self._audience is not None},
            )
            return self._create_verified_token(claims), None
        except InvalidTokenError as exc:
            logger.debug("JWT validation failed: %s", exc)
            return None, str(exc)
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for unknown JWT validation errors
            logger.warning("Unexpected JWT verification error: %s", exc)
            return None, f"Authentication failed: {exc}"

    def _create_verified_token(self, claims: Dict[str, Any]) -> VerifiedClerkToken:
        """Create VerifiedClerkToken from JWT claims."""
        return VerifiedClerkToken(
            user_id=self._extract_user_id(claims),
            email=claims.get("email"),
            first_name=claims.get("first_name"),
            last_name=claims.get("last_name"),
            claims=claims,
        )

    @staticmethod
    def _extract_user_id(claims: Dict[str, Any]) -> str:
        """
        Extract user ID from JWT claims.

        Checks multiple claim keys in priority order:
        1. sub (standard JWT subject)
        2. user_id (custom claim)
        3. sid (session ID fallback)
        """
        for key in ("sub", "user_id", "sid"):
            if claims.get(key):
                return str(claims[key])
        raise InvalidTokenError("Token missing subject identifier")


class ClerkOpaqueTokenVerifier:
    """
    Verifies Clerk-issued OAuth opaque tokens via the /oauth/userinfo endpoint.

    This is used for MCP clients (like ChatGPT) that receive opaque access tokens
    rather than JWTs. Unlike JWTs, opaque tokens cannot be validated locally
    and require an HTTP call to Clerk's userinfo endpoint.

    Example:
        verifier = ClerkOpaqueTokenVerifier(
            userinfo_url="https://clerk.your-domain.com/oauth/userinfo",
        )
        result = await verifier.verify_token("opaque-access-token")
        if result:
            print(f"Authenticated user: {result.user_id}")
    """

    def __init__(
        self,
        userinfo_url: str,
        timeout: float = 10.0,
    ) -> None:
        """
        Initialize the opaque token verifier.

        Args:
            userinfo_url: Clerk userinfo endpoint URL
                         (e.g., https://clerk.your-domain.com/oauth/userinfo)
            timeout: HTTP request timeout in seconds (default: 10.0)
        """
        if not userinfo_url:
            raise ValueError("userinfo_url is required for ClerkOpaqueTokenVerifier")

        self._userinfo_url = userinfo_url
        self._timeout = timeout

    async def verify_token(self, token: str) -> Optional[VerifiedClerkToken]:
        """
        Verify an opaque access token by calling Clerk's userinfo endpoint.

        Args:
            token: The opaque access token string (without "Bearer " prefix)

        Returns:
            VerifiedClerkToken if verification succeeds, None if it fails
        """
        result, _ = await self.verify_token_with_error(token)
        return result

    async def verify_token_with_error(
        self, token: str
    ) -> tuple[Optional[VerifiedClerkToken], Optional[str]]:
        """
        Verify an opaque access token and return both result and error message.

        Args:
            token: The opaque access token string (without "Bearer " prefix)

        Returns:
            Tuple of (VerifiedClerkToken or None, error_message or None)
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    self._userinfo_url,
                    headers={"Authorization": f"Bearer {token}"},
                )

                # Map specific status codes to error messages
                status_error_map = {
                    401: "Token is invalid or expired",
                    403: "Insufficient permissions"
                }
                if response.status_code in status_error_map:
                    return None, status_error_map[response.status_code]

                response.raise_for_status()
                userinfo = response.json()

            return self._create_verified_token(userinfo), None

        except httpx.HTTPStatusError as exc:
            logger.debug("Userinfo request failed: %s %s", exc.response.status_code, exc.response.text[:200])
            return None, f"Token validation failed: {exc.response.status_code}"
        except httpx.TimeoutException:
            logger.warning("Userinfo request timed out")
            return None, "Token validation timed out"
        except httpx.RequestError as exc:
            logger.warning("Userinfo request error: %s", exc)
            return None, f"Token validation error: {exc}"
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for unknown validation errors
            logger.warning("Unexpected userinfo error: %s", exc)
            return None, f"Authentication failed: {exc}"

    def _create_verified_token(self, userinfo: Dict[str, Any]) -> VerifiedClerkToken:
        """
        Create VerifiedClerkToken from userinfo response.

        Clerk userinfo response format:
        {
            "user_id": "user_xxx",
            "email": "user@example.com",
            "email_verified": true,
            "name": "Full Name",
            "given_name": "First",
            "family_name": "Last",
            "picture": "https://...",
            "public_metadata": {...},
            "private_metadata": {...}
        }
        """
        # Extract user_id - check both 'user_id' and 'sub' (standard OIDC claim)
        user_id = userinfo.get("user_id") or userinfo.get("sub")
        if not user_id:
            raise ValueError("Userinfo response missing user_id")

        return VerifiedClerkToken(
            user_id=user_id,
            email=userinfo.get("email"),
            first_name=userinfo.get("given_name"),
            last_name=userinfo.get("family_name"),
            claims=userinfo,  # Store full userinfo as claims for access to metadata
        )
