from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pytz
from croniter import croniter

from seer.core.triggers.polling.adapters.base import (
    PollAdapter,
    PollAdapterError,
    PollContext,
    PolledEvent,
    PollResult,
    register_adapter,
)
from seer.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEZONE = "UTC"


class CronScheduleAdapter(PollAdapter):
    """
    Generate scheduled events based on cron expressions.

    Unlike traditional poll adapters, this doesn't query external APIs.
    It computes the next execution time and generates synthetic events
    when the schedule is due.
    """

    trigger_key = "schedule.cron"

    async def bootstrap_cursor(self, ctx: PollContext) -> Dict[str, Any]:
        """Initialize cursor with the current time as the last execution."""
        config = ctx.subscription.provider_config or {}
        cron_expr = config.get("cron_expression")
        tz_name = config.get("timezone", DEFAULT_TIMEZONE)

        if not cron_expr:
            raise PollAdapterError(
                "Missing cron_expression in provider_config",
                permanent=True,
                detail={"provider_config": config},
            )

        # Validate timezone
        try:
            tz = pytz.timezone(tz_name)
        except Exception as exc:
            raise PollAdapterError(
                f"Invalid timezone: {tz_name}",
                permanent=True,
                detail={"timezone": tz_name, "error": str(exc)},
            ) from exc

        # Validate cron expression
        try:
            now_utc = datetime.now(timezone.utc)
            now_tz = now_utc.astimezone(tz)
            iter_obj = croniter(cron_expr, now_tz)
            # Test that we can get next execution
            iter_obj.get_next(datetime)
        except Exception as exc:
            raise PollAdapterError(
                f"Invalid cron expression: {cron_expr}",
                permanent=True,
                detail={"cron_expression": cron_expr, "error": str(exc)},
            ) from exc

        # Start from now - we don't want to backfill missed runs
        return {
            "last_execution_utc": now_utc.isoformat(),
            "cron_expression": cron_expr,
            "timezone": tz_name,
        }

    async def poll(self, ctx: PollContext, cursor: Dict[str, Any]) -> PollResult:
        """
        Check if it's time to fire based on the cron schedule.

        Returns:
        - Empty events list if not yet due
        - Single event if schedule has triggered
        - Updates cursor with next execution time
        """
        config = ctx.subscription.provider_config or {}
        cron_expr = cursor.get("cron_expression") or config.get("cron_expression")
        tz_name = cursor.get("timezone") or config.get("timezone", DEFAULT_TIMEZONE)
        last_exec_str = cursor.get("last_execution_utc")

        if not cron_expr:
            raise PollAdapterError(
                "Missing cron_expression",
                permanent=True,
                detail={"cursor": cursor, "config": config},
            )

        try:
            tz = pytz.timezone(tz_name)
        except Exception as exc:
            raise PollAdapterError(
                f"Invalid timezone: {tz_name}",
                permanent=True,
                detail={"timezone": tz_name},
            ) from exc

        now_utc = datetime.now(timezone.utc)

        # Determine the reference point for next execution
        if last_exec_str:
            try:
                last_exec_utc = datetime.fromisoformat(last_exec_str)
            except Exception:
                last_exec_utc = now_utc
        else:
            last_exec_utc = now_utc

        # Convert to target timezone for cron calculation
        last_exec_tz = last_exec_utc.astimezone(tz)

        try:
            iter_obj = croniter(cron_expr, last_exec_tz)
            next_exec_tz = iter_obj.get_next(datetime)
            next_exec_utc = next_exec_tz.astimezone(timezone.utc)
        except Exception as exc:
            raise PollAdapterError(
                f"Failed to compute next execution: {exc}",
                permanent=True,
                detail={"cron_expression": cron_expr, "error": str(exc)},
            ) from exc

        # Check if the scheduled time has passed
        if now_utc < next_exec_utc:
            # Not yet time - return empty result but hint when to wake up
            seconds_until_next = max(1, int((next_exec_utc - now_utc).total_seconds()))
            logger.info(
                f"""Cron schedule not yet due:
                    subscription_id: {ctx.subscription.id},
                    next_execution: {next_exec_utc.isoformat()},
                    seconds_until_next: {seconds_until_next},
                """,
            )
            return PollResult(
                events=[],
                cursor={
                    "last_execution_utc": last_exec_utc.isoformat(),
                    "cron_expression": cron_expr,
                    "timezone": tz_name,
                },
                has_more=False,
                rate_limit_hint=seconds_until_next,
            )

        # Time to fire!
        logger.info(
            f"Cron schedule triggered: {cron_expr}",
            extra={
                "subscription_id": ctx.subscription.id,
                "scheduled_time": next_exec_utc.isoformat(),
                "actual_time": now_utc.isoformat(),
            },
        )

        payload = {
            "scheduled_time": next_exec_utc.isoformat(),
            "actual_time": now_utc.isoformat(),
            "cron_expression": cron_expr,
            "timezone": tz_name,
        }

        event = PolledEvent(
            payload=payload,
            raw=None,
            provider_event_id=None,  # Will use event_hash for deduplication
            occurred_at=next_exec_utc,
        )

        # Update cursor to the scheduled execution time (not actual time)
        # This ensures we don't skip or duplicate executions
        new_cursor = {
            "last_execution_utc": next_exec_utc.isoformat(),
            "cron_expression": cron_expr,
            "timezone": tz_name,
        }

        # Hint when to poll again by computing the following execution
        try:
            next_after_tz = croniter(cron_expr, next_exec_tz).get_next(datetime)
            next_after_utc = next_after_tz.astimezone(timezone.utc)
            seconds_until_following = max(1, int((next_after_utc - now_utc).total_seconds()))
        except Exception:
            seconds_until_following = None

        return PollResult(
            events=[event],
            cursor=new_cursor,
            has_more=False,
            rate_limit_hint=seconds_until_following,
        )


register_adapter(CronScheduleAdapter())
