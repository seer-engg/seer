from __future__ import annotations

import os
import ssl
from typing import Optional

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


redis_url = _resolve_redis_url()

# Enable TLS/SSL for rediss:// URLs
ssl_kwargs = {}
if redis_url.startswith("rediss://"):
    ssl_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    logger.info("TLS/SSL enabled for Valkey/Redis connections")

result_backend = RedisAsyncResultBackend(redis_url=redis_url, **ssl_kwargs)
broker = RedisStreamBroker(url=redis_url, **ssl_kwargs).with_result_backend(result_backend)

__all__ = ["broker", "redis_url", "result_backend"]
