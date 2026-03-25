"""
HITL Escalation Scheduler — runs check_hitl_escalations every 5 minutes.

Follows the same pattern as TriggerPollScheduler but simpler.
"""
from __future__ import annotations

import asyncio

from seer.logger import get_logger

logger = get_logger(__name__)

DEFAULT_INTERVAL_SECONDS = 300  # 5 minutes


class HitlEscalationScheduler:
    """Periodically checks for HITL interrupts needing escalation."""

    def __init__(self, interval_seconds: float = DEFAULT_INTERVAL_SECONDS):
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        logger.info("HITL escalation scheduler started (interval=%ss)", self._interval)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("HITL escalation scheduler stopped")

    async def _loop(self) -> None:
        from seer.worker.tasks.hitl_escalation import check_hitl_escalations  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

        while True:
            try:
                result = await check_hitl_escalations()
                if result.get("reminders_sent") or result.get("calls_made"):
                    logger.info("HITL escalation tick: %s", result)
            except Exception:  # pylint: disable=broad-exception-caught  # Reason: scheduler must not crash on transient errors
                logger.exception("HITL escalation tick failed")

            await asyncio.sleep(self._interval)
