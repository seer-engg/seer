from __future__ import annotations

from worker.broker import broker
from shared.config import config
from shared.logger import get_logger

logger = get_logger(__name__)


@broker.task
async def poll_triggers_once() -> None:
    """Run a single TriggerPollEngine tick. Useful for ad-hoc debugging."""
    from api.triggers.polling.engine import TriggerPollEngine  # local import

    engine = TriggerPollEngine(
        max_batch_size=config.trigger_poller_max_batch_size,
        lock_timeout_seconds=config.trigger_poller_lock_timeout_seconds,
    )
    logger.info("Running ad-hoc trigger poll tick")
    await engine.tick()


__all__ = ["poll_triggers_once"]


