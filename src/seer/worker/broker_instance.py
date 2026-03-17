from __future__ import annotations

import os
import socket
import ssl
import sys
from typing import Any, Dict, Optional

from taskiq_redis import RedisAsyncResultBackend, RedisStreamBroker

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


def _resolve_redis_url() -> str:
    """Prefer config.redis_url but fall back to REDIS_URL or localhost.

    Note: We use Valkey (Redis-compatible) in production, but maintain
    'redis_url' naming for backward compatibility with existing configs.
    """
    configured: Optional[str] = getattr(config, "redis_url", None)
    if configured:
        return configured
    env_value = os.getenv("REDIS_URL")
    if env_value:
        return env_value
    return "redis://localhost:6379/0"


def _build_keepalive_options() -> Dict[int, int]:
    """Build TCP keepalive options for the current platform.

    These settings ensure idle connections are detected before AWS/cloud
    infrastructure terminates them (typically 10-15 min idle timeout).
    """
    options: Dict[int, int] = {}

    # Linux-specific TCP keepalive constants
    if sys.platform == "linux":
        # TCP_KEEPIDLE: Start sending keepalive probes after 60s of idle
        options[socket.TCP_KEEPIDLE] = 60
        # TCP_KEEPINTVL: Send keepalive probes every 15s
        options[socket.TCP_KEEPINTVL] = 15
        # TCP_KEEPCNT: Consider connection dead after 3 failed probes
        options[socket.TCP_KEEPCNT] = 3

    return options


def _build_connection_kwargs() -> Dict[str, Any]:
    """Build connection kwargs for Redis/Valkey with resilience settings.

    Returns connection parameters that prevent silent connection termination
    by cloud infrastructure (AWS NAT Gateway, Valkey Serverless, etc.).
    """
    kwargs: Dict[str, Any] = {}

    # Socket timeouts - prevent infinite hangs
    kwargs["socket_timeout"] = config.redis_socket_timeout
    kwargs["socket_connect_timeout"] = config.redis_socket_connect_timeout

    # TCP keepalive - detect dead connections proactively
    if config.redis_socket_keepalive:
        kwargs["socket_keepalive"] = True
        keepalive_options = _build_keepalive_options()
        if keepalive_options:
            kwargs["socket_keepalive_options"] = keepalive_options

    # Health checks - ping connections before use
    kwargs["health_check_interval"] = config.redis_health_check_interval

    # Retry on timeout - auto-retry transient failures
    kwargs["retry_on_timeout"] = True

    return kwargs


def _warn_if_idle_timeout_conflicts() -> None:
    """Warn when broker reclaim timeout can preempt the chat runtime timeout."""
    nexus_timeout_ms = int(config.nexus_chat_timeout_seconds) * 1000
    if config.redis_stream_idle_timeout_ms <= nexus_timeout_ms:
        logger.warning(
            "Redis stream idle timeout should exceed Nexus chat timeout",
            extra={
                "redis_stream_idle_timeout_ms": config.redis_stream_idle_timeout_ms,
                "nexus_chat_timeout_seconds": config.nexus_chat_timeout_seconds,
                "nexus_chat_timeout_ms": nexus_timeout_ms,
            },
        )


def create_redis_client(*, decode_responses: bool = True):
    """Create an async Redis client with the worker's connection settings."""
    import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel # Reason: Optional dependency used lazily at runtime

    return aioredis.from_url(
        redis_url,
        decode_responses=decode_responses,
        **connection_kwargs,
    )


redis_url = _resolve_redis_url()

# Build connection kwargs with resilience settings
connection_kwargs = _build_connection_kwargs()
_warn_if_idle_timeout_conflicts()

# Enable TLS/SSL for rediss:// URLs
if redis_url.startswith("rediss://"):
    connection_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    logger.info("TLS/SSL enabled for Valkey/Redis connections")

logger.info(
    "Redis broker configured with keepalive=%s, socket_timeout=%s, health_check_interval=%s",
    connection_kwargs.get("socket_keepalive", False),
    connection_kwargs.get("socket_timeout"),
    connection_kwargs.get("health_check_interval"),
)

result_backend = RedisAsyncResultBackend(redis_url=redis_url, **connection_kwargs)
broker = RedisStreamBroker(
    url=redis_url,
    max_connection_pool_size=config.redis_max_connections,
    idle_timeout=config.redis_stream_idle_timeout_ms,
    **connection_kwargs,
).with_result_backend(result_backend)

__all__ = ["broker", "redis_url", "result_backend", "create_redis_client"]
