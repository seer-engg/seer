from __future__ import annotations

import os
from typing import Optional

from taskiq.events import TaskiqEvents
from taskiq.state import TaskiqState
from taskiq_redis import RedisAsyncResultBackend, RedisStreamBroker

from seer.config import config
from seer.database import close_db, init_db
from seer.logger import get_logger
from seer.core.triggers.polling import TriggerPollScheduler  # lazy import
from seer.analytics import analytics  # PostHog analytics

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

_poll_scheduler = None  # pylint: disable=invalid-name


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def _on_worker_startup(_: TaskiqState) -> None:
    """Initialize shared resources before processing tasks."""
    # pylint: disable=import-outside-toplevel,global-statement
    from seer.worker.trigger_dispatcher import dispatch_trigger_event  # noqa: F401

    global _poll_scheduler

    logger.info("Initializing Taskiq worker")
    await init_db()

    # Initialize PostHog for worker analytics
    analytics.initialize()

    if config.trigger_poller_enabled:
        logger.info("Starting trigger poll scheduler in worker")
        _poll_scheduler = TriggerPollScheduler(
            interval_seconds=config.trigger_poller_interval_seconds,
            max_batch_size=config.trigger_poller_max_batch_size,
            lock_timeout_seconds=config.trigger_poller_lock_timeout_seconds,
            trigger_event_dispatcher=dispatch_trigger_event,
        )
        await _poll_scheduler.start()
    else:
        logger.info("Trigger poller disabled via configuration")


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def _on_worker_shutdown(_: TaskiqState) -> None:
    """Clean up background services when worker exits."""
    # pylint: disable=import-outside-toplevel,global-statement
    from seer.analytics import analytics  # PostHog analytics
    global _poll_scheduler

    if _poll_scheduler:
        logger.info("Stopping trigger poll scheduler")
        await _poll_scheduler.stop()
        _poll_scheduler = None

    # Flush and shutdown PostHog before closing DB
    analytics.flush()
    analytics.shutdown()

    await close_db()
    logger.info("Taskiq worker shutdown complete")


# Import task modules to register with broker
# pylint: disable=wrong-import-position,unused-import
from seer.worker.tasks import workflows, triggers, polling, stripe  # noqa: F401

__all__ = ["broker"]
