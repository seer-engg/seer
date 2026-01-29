from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
from seer.database.subscription_models import BillingProfile
from seer.logger import get_logger
from seer.observability import (
    PaymentMethodRequiredError,
    TrialExpiredError,
    get_account_age_days,
    is_trial_expired,
)

logger = get_logger("api.middleware.auth")

PAYMENT_METHOD_GRANDFATHER_CUTOFF = datetime(2026, 2, 1, tzinfo=timezone.utc)


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

        error_response = await self._authenticate_and_validate(request)
        if error_response:
            return error_response

        return await call_next(request)

    async def _authenticate_and_validate(self, request: Request) -> Optional[JSONResponse]:
        """Authenticate user and validate billing requirements."""
        token = self._extract_token(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        claims, error = self._decode_and_validate_jwt(token)
        if error:
            return error

        auth_user = AuthenticatedUser(
            user_id=self._extract_user_id(claims),
            email=claims.get("email"),
            first_name=claims.get("first_name"),
            last_name=claims.get("last_name"),
            claims=claims,
        )

        db_user, error = await self._create_or_get_user(auth_user, request)
        if error:
            return error

        error = await self._check_trial_expiration(db_user)
        if error:
            return error

        error = await self._check_payment_method_required(db_user)
        if error:
            return error

        request.state.user = auth_user
        request.state.db_user = db_user
        return None

    def _decode_and_validate_jwt(self, token: str) -> tuple[Optional[Dict], Optional[JSONResponse]]:
        """Decode and validate JWT token."""
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
            return claims, None
        except InvalidTokenError as exc:
            return None, JSONResponse(status_code=401, content={"detail": str(exc)})
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for unknown JWT validation errors to prevent auth bypass
            return None, JSONResponse(
                status_code=401,
                content={"detail": f"Authentication failed: {exc}"},
            )

    async def _create_or_get_user(self, auth_user: AuthenticatedUser, request: Request) -> tuple[Optional[User], Optional[JSONResponse]]:
        """Create or retrieve user from database."""
        try:
            signup_source = request.query_params.get("signup_source")
            db_user = await User.get_or_create_from_auth(auth_user, signup_source=signup_source)
            return db_user, None
        except Exception:  # pylint: disable=broad-exception-caught # Reason: Defensive catch for database errors during user creation
            logger.exception("Failed to persist authenticated user")
            return None, JSONResponse(
                status_code=500,
                content={"detail": "Unable to persist authenticated user"},
            )

    async def _check_trial_expiration(self, db_user: User) -> Optional[JSONResponse]:
        """Check if user's trial has expired."""
        if await is_trial_expired(db_user):
            days_since_signup = await get_account_age_days(db_user)
            error = TrialExpiredError(days_since_signup=days_since_signup)
            return JSONResponse(
                status_code=402,
                content=error.to_dict(),
            )
        return None

    async def _check_payment_method_required(self, db_user: User) -> Optional[JSONResponse]:
        """Check if payment method is required for new users."""
        if not config.is_self_hosted:
            if db_user.created_at >= PAYMENT_METHOD_GRANDFATHER_CUTOFF:
                billing_profile = await BillingProfile.get_or_none(owner_user=db_user)

                if not billing_profile or not billing_profile.payment_method_on_file:
                    error = PaymentMethodRequiredError()
                    return JSONResponse(
                        status_code=402,
                        content=error.to_dict(),
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
