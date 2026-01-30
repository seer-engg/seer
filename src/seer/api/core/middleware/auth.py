from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from seer.api.core.middleware.path_allowlist import is_public_path
from seer.config import config
from seer.database import User
from seer.database.models import UserSettings
from seer.database.subscription_models import BillingProfile
from seer.logger import get_logger
from seer.observability import (
    TrialExpiredError,
    get_account_age_days,
    is_trial_expired,
)

logger = get_logger("api.middleware.auth")


@dataclass
class AuthenticatedUser:
    """Represents the authenticated Clerk user attached to a request."""

    user_id: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    claims: Dict[str, Any]


class ClerkAuthMiddleware(BaseHTTPMiddleware):
    """Verifies Clerk-issued bearer tokens and attaches the decoded user to the request."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        jwks_url: str,
        issuer: str,
        audience: Optional[Sequence[str]] = None,
        allow_unauthenticated_paths: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(app)
        if not jwks_url:
            raise ValueError("jwks_url is required for ClerkAuthMiddleware")
        if not issuer:
            raise ValueError("issuer is required for ClerkAuthMiddleware")

        self._jwks_client = PyJWKClient(jwks_url)
        self._issuer = issuer
        self._audience = list(audience) if audience else None
        self._extra_allowed_paths = set(allow_unauthenticated_paths or [])

    async def dispatch(self, request: Request, call_next):
        request.state.user = None
        request.state.db_user = None
        if self._should_skip(request):
            return await call_next(request)

        token = self._extract_token(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        claims = self._verify_token(token)
        if isinstance(claims, JSONResponse):
            return claims

        auth_user = self._create_auth_user(claims)
        db_user = await self._get_or_create_db_user(request, auth_user)
        if isinstance(db_user, JSONResponse):
            return db_user

        gate_response = await self._check_access_gates(request, db_user)
        if gate_response:
            return gate_response

        request.state.user = auth_user
        request.state.db_user = db_user
        return await call_next(request)

    def _verify_token(self, token: str) -> Dict[str, Any] | JSONResponse:
        """Verify JWT token and return claims or error response."""
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
            return claims
        except InvalidTokenError as exc:
            return JSONResponse(status_code=401, content={"detail": str(exc)})
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for unknown JWT validation errors to prevent auth bypass
            return JSONResponse(
                status_code=401,
                content={"detail": f"Authentication failed: {exc}"},
            )

    def _create_auth_user(self, claims: Dict[str, Any]) -> AuthenticatedUser:
        """Create AuthenticatedUser from JWT claims."""
        return AuthenticatedUser(
            user_id=self._extract_user_id(claims),
            email=claims.get("email"),
            first_name=claims.get("first_name"),
            last_name=claims.get("last_name"),
            claims=claims,
        )

    async def _get_or_create_db_user(self, request: Request, auth_user: AuthenticatedUser) -> User | JSONResponse:
        """Get or create database user from authenticated user."""
        try:
            signup_source = request.query_params.get("signup_source")
            return await User.get_or_create_from_auth(auth_user, signup_source=signup_source)
        except Exception:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for database errors during user creation
            logger.exception("Failed to persist authenticated user")
            return JSONResponse(
                status_code=500,
                content={"detail": "Unable to persist authenticated user"},
            )

    async def _check_access_gates(self, request: Request, db_user: User) -> Optional[JSONResponse]:
        """Check trial expiry and payment method gates. Returns error response or None."""
        # Phase 2: Account Day Limit Gate
        if await is_trial_expired(db_user):
            days_since_signup = await get_account_age_days(db_user)
            error = TrialExpiredError(days_since_signup=days_since_signup)
            return JSONResponse(
                status_code=402,
                content=error.to_dict(),
            )

        # Phase 3: Payment Method Gate (Cloud only)
        if not config.is_self_hosted:
            payment_gate_response = await self._check_payment_method_gate(request, db_user)
            if payment_gate_response:
                return payment_gate_response

        return None

    async def _check_payment_method_gate(self, request: Request, db_user: User) -> Optional[JSONResponse]:
        """Check if payment method is required for this request."""
        path = request.scope.get("path") or request.url.path

        # Allow onboarding-related endpoints and settings API
        if path.startswith("/api/subscriptions") or path == "/api/users/me/settings":
            return None

        billing_profile = await BillingProfile.get_or_none(owner_user=db_user)

        # Require payment method unless they're still in onboarding
        if not billing_profile or not billing_profile.has_payment_method:
            settings = await UserSettings.get_or_none(user=db_user)
            onboarding_complete = settings and settings.preferences.get("onboarding", {}).get("completed", False)

            if onboarding_complete:
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "payment_method_required",
                        "message": "Payment method required to access this resource",
                        "requires_payment_method": True
                    }
                )

        return None

    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from Authorization header or query parameter."""
        # Check Authorization header first
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.removeprefix("Bearer ").strip()
            if token:
                return token

        # Fall back to query parameter (for OAuth redirect flows)
        token = request.query_params.get("token")
        if token:
            return token

        return None

    def _should_skip(self, request: Request) -> bool:
        if request.method == "OPTIONS":
            return True

        path = request.scope.get("path") or request.url.path
        return is_public_path(path, self._extra_allowed_paths)

    @staticmethod
    def _extract_user_id(claims: Dict[str, Any]) -> str:
        for key in ("sub", "user_id", "sid"):
            if claims.get(key):
                return str(claims[key])
        raise InvalidTokenError("Token missing subject identifier")


class TokenDecodeWithoutValidationMiddleware(BaseHTTPMiddleware):
    """Decodes a JWT token without validating signature. Useful for development/testing."""

    async def dispatch(self, request: Request, call_next):
        request.state.user = None
        request.state.db_user = None

        # Skip auth for OPTIONS requests and OAuth callbacks
        if self._should_skip(request):
            return await call_next(request)

        # Try Authorization header first, then fall back to query param (for OAuth redirects)
        token = self._extract_token(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        try:
            # Decode without signature verification
            claims = jwt.decode(token, options={"verify_signature": False})
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for malformed JWT tokens in development mode
            return JSONResponse(
                status_code=401,
                content={"detail": f"Failed to decode User: {exc}"},
            )

        user_id = self._extract_user_id(claims)

        auth_user = AuthenticatedUser(
            user_id=user_id,
            email=claims.get("email"),
            first_name=claims.get("first_name"),
            last_name=claims.get("last_name"),
            claims=claims,
        )

        try:
            # Capture signup_source from query params (for new user signups)
            signup_source = request.query_params.get("signup_source")
            db_user = await User.get_or_create_from_auth(auth_user, signup_source=signup_source)
        except Exception:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for database errors during user creation in development mode
            logger.exception("Failed to persist user from decoded token")
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to persist user from decoded token"},
            )

        request.state.user = auth_user
        request.state.db_user = db_user
        return await call_next(request)

    def _should_skip(self, request: Request) -> bool:
        """Skip auth for OPTIONS requests, health checks, and OAuth callbacks."""
        if request.method == "OPTIONS":
            return True

        path = request.scope.get("path") or request.url.path
        return is_public_path(path, include_docs=True)

    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from Authorization header or query parameter."""
        # Check Authorization header first
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.removeprefix("Bearer ").strip()
            if token:
                return token

        # Fall back to query parameter (for OAuth redirect flows)
        token = request.query_params.get("token")
        if token:
            return token

        return None

    @staticmethod
    def _extract_user_id(claims: Dict[str, Any]) -> Optional[str]:
        for key in ("sub", "user_id", "sid"):
            if claims.get(key):
                return str(claims[key])
        return None
