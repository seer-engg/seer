from __future__ import annotations

import asyncio

from taskiq.events import TaskiqEvents
from taskiq.state import TaskiqState

from seer.config import config
from seer.database import close_db, init_db
from seer.logger import get_logger
from seer.core.triggers.polling import TriggerPollScheduler  # lazy import
from seer.worker.broker_instance import broker
from seer.utilities.ml_flow import _ensure_mlflow_autologging
from seer.utilities.langfuse_tracing import get_nexus_langfuse_callbacks, get_workflow_langfuse_callbacks

if config.mlflow_enabled:
    _ensure_mlflow_autologging()

logger = get_logger(__name__)

if config.langfuse_enabled:
    _nexus_callbacks = get_nexus_langfuse_callbacks()
    _workflow_callbacks = get_workflow_langfuse_callbacks()
    if _nexus_callbacks or _workflow_callbacks:
        _projects = []
        if _nexus_callbacks:
            _projects.append("nexus")
        if _workflow_callbacks:
            _projects.append("workflow")
        logger.info("Langfuse tracing enabled for worker (projects: %s)", ", ".join(_projects))
    else:
        logger.warning("Langfuse enabled but no projects configured")

_poll_scheduler = None  # pylint: disable=invalid-name


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def _on_worker_startup(_: TaskiqState) -> None:
    """Initialize shared resources before processing tasks."""
    # pylint: disable=import-outside-toplevel,global-statement
    from seer.worker.trigger_dispatcher import dispatch_trigger_event  # noqa: F401
    from seer.core.event_loop import set_main_event_loop

    global _poll_scheduler

    # Capture main event loop for cross-thread async operations (same as API)
    set_main_event_loop(asyncio.get_running_loop())
    logger.info("Main event loop captured for cross-thread scheduling")

    logger.info("Initializing Taskiq worker")
    await init_db()

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
    # pylint: disable=global-statement
    global _poll_scheduler

    if _poll_scheduler:
        logger.info("Stopping trigger poll scheduler")
        await _poll_scheduler.stop()
        _poll_scheduler = None

    await close_db()
    logger.info("Taskiq worker shutdown complete")


# Import task modules to register with broker
# pylint: disable=wrong-import-position,unused-import
from seer.worker.tasks import workflows, triggers, polling, stripe, chat  # noqa: F401

__all__ = ["broker"]
