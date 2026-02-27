# pylint: disable=import-outside-toplevel,too-many-arguments,too-many-positional-arguments
# Reason: redis.asyncio import is lazy-loaded for optional dependency; middleware __init__ has many cookie config params
"""
Redis-backed session middleware for multi-worker OAuth flows.

This middleware stores session data in Redis, enabling OAuth flows (especially PKCE)
to work correctly when the /connect and /callback requests hit different workers.

Starlette's default SessionMiddleware uses signed cookies, which work for single-server
setups but fail in distributed deployments because:
1. OAuth state must persist between authorize and callback
2. PKCE code_verifier must be stored server-side (not in URL state for security)
3. Authlib's authorize_access_token() expects session-stored state

By storing sessions in Redis:
- Any worker can handle the callback and retrieve the session
- PKCE secrets are kept server-side (not exposed in URLs)
- Session data expires automatically (TTL-based cleanup)
"""
import json
import secrets
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from seer.logger import get_logger

logger = get_logger(__name__)

# Session cookie name
SESSION_COOKIE_NAME = "seer_session_id"


class RedisSessionBackend:
    """
    Redis-backed session storage for multi-worker OAuth.

    Sessions are stored as JSON in Redis with automatic TTL expiration.
    The session ID is stored in a cookie, while the actual data lives in Redis.
    """

    def __init__(self, redis_url: str, session_ttl: int = 600):
        """
        Initialize Redis session backend.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
            session_ttl: Session time-to-live in seconds (default: 10 minutes)
        """
        self.redis_url = redis_url
        self.session_ttl = session_ttl
        self._redis: Optional["redis.asyncio.Redis"] = None

    async def _get_redis(self):
        """Lazily create Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
            )
        return self._redis

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"session:{session_id}"

    async def read(self, session_id: str) -> dict:
        """
        Read session data from Redis.

        Args:
            session_id: The session identifier

        Returns:
            Session data dict, or empty dict if not found
        """
        if not session_id:
            return {}

        redis = await self._get_redis()
        try:
            data = await redis.get(self._session_key(session_id))
            if data:
                return json.loads(data)
        except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError) as exc:
            # Redis connection errors or malformed JSON data
            logger.warning("Failed to read session %s: %s", session_id[:8], exc)

        return {}

    async def write(self, session_id: str, data: dict) -> str:
        """
        Write session data to Redis with TTL.

        Args:
            session_id: The session identifier (or empty to generate new)
            data: Session data to store

        Returns:
            The session ID (existing or newly generated)
        """
        if not session_id:
            session_id = secrets.token_urlsafe(32)

        redis = await self._get_redis()
        try:
            await redis.setex(
                self._session_key(session_id),
                self.session_ttl,
                json.dumps(data),
            )
            logger.debug("Wrote session %s with TTL=%d", session_id[:8], self.session_ttl)
        except (OSError, ConnectionError, TimeoutError, TypeError) as exc:
            # Redis connection errors or non-serializable data
            logger.error("Failed to write session %s: %s", session_id[:8], exc)

        return session_id

    async def delete(self, session_id: str) -> None:
        """
        Delete session from Redis.

        Args:
            session_id: The session identifier to delete
        """
        if not session_id:
            return

        redis = await self._get_redis()
        try:
            await redis.delete(self._session_key(session_id))
            logger.debug("Deleted session %s", session_id[:8])
        except (OSError, ConnectionError, TimeoutError) as exc:
            # Redis connection errors
            logger.warning("Failed to delete session %s: %s", session_id[:8], exc)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


class SessionInterface:
    """
    Dict-like interface for session data that tracks modifications.

    Authlib and Starlette expect request.session to behave like a dict.
    This wrapper tracks when the session is modified so we know to persist it.
    """

    def __init__(self, data: dict):
        self._data = data
        self._modified = False

    def __getitem__(self, key: str):
        return self._data[key]

    def __setitem__(self, key: str, value):
        self._data[key] = value
        self._modified = True

    def __delitem__(self, key: str):
        del self._data[key]
        self._modified = True

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def pop(self, key: str, *args):
        self._modified = True
        return self._data.pop(key, *args)

    def update(self, other: dict):
        self._data.update(other)
        self._modified = True

    def clear(self):
        self._data.clear()
        self._modified = True

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    @property
    def is_modified(self) -> bool:
        return self._modified

    def to_dict(self) -> dict:
        return self._data.copy()


class RedisSessionMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that provides Redis-backed sessions.

    This middleware:
    1. Reads session ID from cookie on request
    2. Loads session data from Redis into request.session
    3. After response, persists modified session data back to Redis
    4. Sets session cookie if new session was created

    Usage:
        app.add_middleware(
            RedisSessionMiddleware,
            redis_url="redis://localhost:6379/0",
            session_ttl=600,
        )

    Then in routes:
        request.session['key'] = 'value'
        value = request.session.get('key')
    """

    def __init__(
        self,
        app: ASGIApp,
        redis_url: str,
        session_ttl: int = 600,
        cookie_name: str = SESSION_COOKIE_NAME,
        cookie_path: str = "/",
        cookie_secure: bool = False,
        cookie_httponly: bool = True,
        cookie_samesite: str = "lax",
    ):
        """
        Initialize Redis session middleware.

        Args:
            app: The ASGI application
            redis_url: Redis connection URL
            session_ttl: Session TTL in seconds (default: 10 minutes)
            cookie_name: Name of the session cookie
            cookie_path: Cookie path
            cookie_secure: Require HTTPS for cookie
            cookie_httponly: Make cookie inaccessible to JavaScript
            cookie_samesite: SameSite cookie policy ("lax", "strict", "none")
        """
        super().__init__(app)
        self.backend = RedisSessionBackend(redis_url, session_ttl)
        self.cookie_name = cookie_name
        self.cookie_path = cookie_path
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite
        self.session_ttl = session_ttl

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request with Redis session."""
        # Get session ID from cookie
        session_id = request.cookies.get(self.cookie_name, "")

        # Load session data from Redis
        session_data = await self.backend.read(session_id)
        session = SessionInterface(session_data)

        # Attach session to request scope (where Starlette/Authlib expect it)
        request.scope["session"] = session

        # Also set as request.state.session for convenience
        request.state.session = session

        # Process request
        response = await call_next(request)

        # Persist session if modified
        if session.is_modified or not session_id:
            new_session_id = await self.backend.write(
                session_id if session_id else "",
                session.to_dict(),
            )

            # Set cookie if session ID changed (new session)
            if new_session_id != session_id:
                self._set_session_cookie(response, new_session_id)

        return response

    def _set_session_cookie(self, response: Response, session_id: str) -> None:
        """Set the session cookie on the response."""
        # Use Response.set_cookie() for proper cookie formatting
        response.set_cookie(
            key=self.cookie_name,
            value=session_id,
            max_age=self.session_ttl,
            path=self.cookie_path,
            secure=self.cookie_secure,
            httponly=self.cookie_httponly,
            samesite=self.cookie_samesite,
        )
        logger.debug(
            "Set session cookie: name=%s, session_id=%s..., samesite=%s, secure=%s",
            self.cookie_name,
            session_id[:8] if session_id else "None",
            self.cookie_samesite,
            self.cookie_secure,
        )
