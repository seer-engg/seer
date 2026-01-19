from __future__ import annotations

import asyncio
from typing import Optional, Callable

# Ensure adapters register themselves.
import seer.core.triggers.polling.adapters  # noqa: F401
from seer.core.triggers.polling.engine import TriggerPollEngine
from seer.logger import get_logger

logger = get_logger(__name__)


class TriggerPollScheduler:
    """Background loop that periodically ticks the poll engine."""

    def __init__(
        self,
        *,
        interval_seconds: int = 5,
        max_batch_size: int = 10,
        lock_timeout_seconds: int = 60,
        trigger_event_dispatcher: Callable,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.engine = TriggerPollEngine(
            trigger_event_dispatcher=trigger_event_dispatcher,
            max_batch_size=max_batch_size,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Trigger poll scheduler started", extra={"worker_id": self.engine.worker_id})

    async def stop(self) -> None:
        if not self._task:
            return
        self._stop_event.set()
        await self._task
        logger.info("Trigger poll scheduler stopped", extra={"worker_id": self.engine.worker_id})

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.engine.tick()
            except Exception:
                logger.exception("Trigger poll engine tick failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue
