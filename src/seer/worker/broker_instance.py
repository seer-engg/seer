from __future__ import annotations

import os
from typing import Optional

from taskiq_redis import RedisAsyncResultBackend, RedisStreamBroker

from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


def _resolve_redis_url() -> str:
    """Prefer config.redis_url but fall back to REDIS_URL or localhost."""
    configured: Optional[str] = getattr(config, "redis_url", None)
    if configured:
        return configured
    env_value = os.getenv("REDIS_URL")
    if env_value:
        return env_value
    return "redis://localhost:6379/0"


redis_url = _resolve_redis_url()
result_backend = RedisAsyncResultBackend(redis_url=redis_url)
broker = RedisStreamBroker(url=redis_url).with_result_backend(result_backend)

__all__ = ["broker", "redis_url", "result_backend"]
