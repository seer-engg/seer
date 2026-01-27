from __future__ import annotations

from seer.worker.broker_instance import broker
from seer.config import config
from seer.logger import get_logger

logger = get_logger(__name__)


@broker.task
async def poll_triggers_once() -> None:
    """Run a single TriggerPollEngine tick. Useful for ad-hoc debugging."""
    from seer.core.triggers.polling.engine import TriggerPollEngine  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports in worker task
    from seer.worker.trigger_dispatcher import dispatch_trigger_event  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports in worker task

    engine = TriggerPollEngine(
        max_batch_size=config.trigger_poller_max_batch_size,
        lock_timeout_seconds=config.trigger_poller_lock_timeout_seconds,
        trigger_event_dispatcher=dispatch_trigger_event
    )
    logger.info("Running ad-hoc trigger poll tick")
    await engine.tick()


__all__ = ["poll_triggers_once"]
