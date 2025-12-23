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

from shared.logger import get_logger

logger = get_logger("api.middleware.auth")
from shared.database.models import User


@dataclass
class AuthenticatedUser:
    """Represents the authenticated Clerk user attached to a request."""

    user_id: str
    email: Optional[str]
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
        self._allowed_paths = set(allow_unauthenticated_paths or [])

    async def dispatch(self, request: Request, call_next):
        request.state.user = None
        request.state.db_user = None
        if self._should_skip(request):
            return await call_next(request)

        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing bearer token"},
            )

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
        except InvalidTokenError as exc:
            return JSONResponse(status_code=401, content={"detail": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            return JSONResponse(
                status_code=401,
                content={"detail": f"Authentication failed: {exc}"},
            )

        auth_user = AuthenticatedUser(
            user_id=self._extract_user_id(claims),
            email=claims.get("email"),
            first_name=claims.get("first_name"),
            last_name=claims.get("last_name"),
            claims=claims,
        )
        try:
            db_user = await User.get_or_create_from_auth(auth_user)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to persist authenticated user")
            return JSONResponse(
                status_code=500,
                content={"detail": "Unable to persist authenticated user"},
            )

        request.state.user = auth_user
        request.state.db_user = db_user
        return await call_next(request)

    def _should_skip(self, request: Request) -> bool:
        if request.method == "OPTIONS":
            return True

        path = request.scope.get("path") or request.url.path
        for allowed in self._allowed_paths:
            if allowed == "/":
                if path == "/":
                    return True
                continue

            normalized = allowed.rstrip("/")
            if path == normalized or path.startswith(f"{normalized}/"):
                return True
        return False

    @staticmethod
    def _extract_user_id(claims: Dict[str, Any]) -> str:
        for key in ("sub", "user_id", "sid"):
            if claims.get(key):
                return str(claims[key])
        raise InvalidTokenError("Token missing subject identifier")



class TokenDecodeWithoutValidationMiddleware(BaseHTTPMiddleware):
    """Decodes a JWT token without validating signature. Useful for development/testing."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        request.state.user = None
        request.state.db_user = None

        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return await call_next(request)

        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            return await call_next(request)

        try:
            # Decode without signature verification
            claims = jwt.decode(token, options={"verify_signature": False})
        except Exception as exc:
            logger.warning(f"Failed to decode token: {exc}")
            return await call_next(request)

        user_id = self._extract_user_id(claims)
        if not user_id:
            return await call_next(request)

        auth_user = AuthenticatedUser(
            user_id=user_id,
            email=claims.get("email"),
            first_name=claims.get("first_name"),
            last_name=claims.get("last_name"),
            claims=claims,
        )

        try:
            db_user = await User.get_or_create_from_auth(auth_user)
        except Exception:
            logger.exception("Failed to persist user from decoded token")
            return await call_next(request)

        request.state.user = auth_user
        request.state.db_user = db_user
        return await call_next(request)

    @staticmethod
    def _extract_user_id(claims: Dict[str, Any]) -> Optional[str]:
        for key in ("sub", "user_id", "sid"):
            if claims.get(key):
                return str(claims[key])
        return None