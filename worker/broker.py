from __future__ import annotations

import os
from typing import Optional

from taskiq.events import TaskiqEvents
from taskiq.state import TaskiqState
from taskiq_redis import RedisAsyncResultBackend, RedisStreamBroker

from shared.config import config
from shared.database import close_db, init_db
from shared.logger import get_logger

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

_poll_scheduler = None


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def _on_worker_startup(_: TaskiqState) -> None:
    """Initialize shared resources before processing tasks."""
    from api.triggers.polling import TriggerPollScheduler  # lazy import

    global _poll_scheduler

    logger.info("Initializing Taskiq worker")
    await init_db()

    if config.trigger_poller_enabled:
        logger.info("Starting trigger poll scheduler in worker")
        _poll_scheduler = TriggerPollScheduler(
            interval_seconds=config.trigger_poller_interval_seconds,
            max_batch_size=config.trigger_poller_max_batch_size,
            lock_timeout_seconds=config.trigger_poller_lock_timeout_seconds,
        )
        await _poll_scheduler.start()
    else:
        logger.info("Trigger poller disabled via configuration")


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def _on_worker_shutdown(_: TaskiqState) -> None:
    """Clean up background services when worker exits."""
    global _poll_scheduler

    if _poll_scheduler:
        logger.info("Stopping trigger poll scheduler")
        await _poll_scheduler.stop()
        _poll_scheduler = None

    await close_db()
    logger.info("Taskiq worker shutdown complete")


__all__ = ["broker"]


