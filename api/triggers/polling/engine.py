from __future__ import annotations
import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import uuid4

from fastapi import HTTPException
from tortoise.expressions import Q
from tortoise.transactions import in_transaction

from api.triggers.polling.adapters.base import (
    PollAdapterError,
    PollContext,
    adapter_registry,
)
from api.triggers.polling.dedupe import compute_event_hash
from api.triggers.services import (
    _build_event_envelope,
    _dispatch_trigger_event,
    _load_trigger_provider,
    _persist_event,
)
from shared.database.workflow_models import TriggerSubscription
from shared.logger import get_logger
from shared.tools.executor import get_oauth_token

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TriggerPollEngine:
    """Coordinates leasing subscriptions and delegating to provider adapters."""

    def __init__(
        self,
        *,
        lock_timeout_seconds: int = 60,
        max_batch_size: int = 10,
    ) -> None:
        self.lock_timeout = timedelta(seconds=lock_timeout_seconds)
        self.max_batch_size = max_batch_size
        self.worker_id = f"poller-{uuid4().hex[:8]}"

    async def tick(self) -> None:
        subscriptions = await self._lease_due_subscriptions(limit=self.max_batch_size)
        if not subscriptions:
            return

        for subscription in subscriptions:
            try:
                await self._process_subscription(subscription)
            except Exception:
                logger.exception(
                    "Failed to process trigger subscription",
                    extra={"subscription_id": subscription.id, "trigger_key": subscription.trigger_key},
                )
                await self._mark_error(
                    subscription,
                    reason="Unhandled poller exception",
                    detail={"worker_id": self.worker_id},
                    delay_seconds=subscription.poll_interval_seconds,
                )

    async def _lease_due_subscriptions(self, *, limit: int) -> List[TriggerSubscription]:
        now = _utcnow()
        async with in_transaction() as conn:
            queryset = (
                TriggerSubscription.filter(
                    enabled=True,
                    next_poll_at__lte=now,
                )
                .exclude(poll_status="disabled")
                .filter(Q(poll_lock_owner__isnull=True) | Q(poll_lock_expires_at__lte=now))
                .order_by("next_poll_at")
                .limit(limit)
                .using_db(conn)
                .select_for_update(skip_locked=True)
            )
            subscriptions = await queryset
            if not subscriptions:
                return []

            lock_expiry = now + self.lock_timeout
            for subscription in subscriptions:
                subscription.poll_lock_owner = self.worker_id
                subscription.poll_lock_expires_at = lock_expiry
                await subscription.save(
                    update_fields=["poll_lock_owner", "poll_lock_expires_at"],
                    using_db=conn,
                )
            return subscriptions

    async def _process_subscription(self, subscription: TriggerSubscription) -> None:
        adapter = adapter_registry.get(subscription.trigger_key)
        if adapter is None:
            logger.error(
                "No poll adapter registered for trigger",
                extra={"subscription_id": subscription.id, "trigger_key": subscription.trigger_key},
            )
            await self._disable_subscription(subscription, reason="missing_adapter")
            return

        if subscription.provider_connection_id is None:
            await self._mark_error(
                subscription,
                reason="missing_provider_connection",
                detail={"trigger_key": subscription.trigger_key},
                delay_seconds=max(subscription.poll_interval_seconds, 60),
            )
            return

        await subscription.fetch_related("user")
        user = subscription.user
        if user is None:
            await self._disable_subscription(subscription, reason="missing_user")
            return

        try:
            connection, access_token = await get_oauth_token(
                user,
                connection_id=str(subscription.provider_connection_id),
            )
        except HTTPException as exc:
            should_disable = exc.status_code in {401, 403, 404}
            if should_disable:
                await self._disable_subscription(
                    subscription,
                    reason="oauth_error",
                    detail={"status_code": exc.status_code, "detail": exc.detail},
                )
            else:
                await self._mark_error(
                    subscription,
                    reason="oauth_error",
                    detail={"status_code": exc.status_code, "detail": exc.detail},
                    delay_seconds=subscription.poll_interval_seconds,
                )
            return

        ctx = PollContext(
            subscription=subscription,
            user=user,
            connection=connection,
            access_token=access_token,
        )

        cursor = subscription.poll_cursor_json or None
        if cursor is None:
            cursor = await adapter.bootstrap_cursor(ctx)

        try:
            result = await adapter.poll(ctx, cursor)
        except PollAdapterError as exc:
            if exc.permanent:
                await self._disable_subscription(
                    subscription, reason="adapter_permanent_error", detail=exc.detail
                )
                return
            backoff = exc.backoff_seconds or min(subscription.poll_interval_seconds * 2, 600)
            await self._mark_backoff(
                subscription,
                reason="adapter_error",
                detail=exc.detail or {"message": str(exc)},
                backoff_seconds=backoff,
            )
            return

        await self._handle_events(subscription, result.events)
        await self._mark_success(
            subscription,
            cursor=result.cursor,
            has_more=result.has_more,
            rate_limit_hint=result.rate_limit_hint,
        )

    async def _handle_events(self, subscription: TriggerSubscription, events) -> None:
        if not events:
            return
        provider = _load_trigger_provider(subscription.trigger_key)
        for polled in events:
            envelope = _build_event_envelope(
                trigger_key=subscription.trigger_key,
                provider=provider,
                provider_connection_id=subscription.provider_connection_id,
                payload=polled.payload,
                raw=polled.raw,
                occurred_at=polled.occurred_at,
            )

            provider_event_id = polled.provider_event_id
            event_hash = None
            if provider_event_id is None:
                event_hash = compute_event_hash(
                    trigger_key=subscription.trigger_key,
                    provider_connection_id=subscription.provider_connection_id,
                    envelope=envelope,
                )

            event, created = await _persist_event(
                subscription=subscription,
                envelope=envelope,
                provider_event_id=provider_event_id,
                event_hash=event_hash,
                raw=polled.raw,
            )
            if created:
                await _dispatch_trigger_event(subscription, event, envelope)

    async def _mark_success(
        self,
        subscription: TriggerSubscription,
        *,
        cursor,
        has_more: bool,
        rate_limit_hint: Optional[int],
    ) -> None:
        interval = subscription.poll_interval_seconds
        jitter_window = max(1, min(10, int(interval * 0.1)))
        jitter = random.uniform(0, jitter_window)
        if has_more:
            next_poll = _utcnow() + timedelta(seconds=1)
        elif rate_limit_hint:
            next_poll = _utcnow() + timedelta(seconds=rate_limit_hint)
        else:
            next_poll = _utcnow() + timedelta(seconds=interval + jitter)

        subscription.poll_cursor_json = cursor
        subscription.poll_status = "ok"
        subscription.poll_error_json = None
        subscription.poll_backoff_seconds = 0
        subscription.next_poll_at = next_poll
        subscription.poll_lock_owner = None
        subscription.poll_lock_expires_at = None
        await subscription.save(
            update_fields=[
                "poll_cursor_json",
                "poll_status",
                "poll_error_json",
                "poll_backoff_seconds",
                "next_poll_at",
                "poll_lock_owner",
                "poll_lock_expires_at",
            ]
        )

    async def _mark_backoff(
        self,
        subscription: TriggerSubscription,
        *,
        reason: str,
        detail: Optional[dict],
        backoff_seconds: int,
    ) -> None:
        next_poll = _utcnow() + timedelta(seconds=backoff_seconds)
        subscription.poll_status = "backoff"
        subscription.poll_error_json = {"reason": reason, "detail": detail}
        subscription.poll_backoff_seconds = backoff_seconds
        subscription.next_poll_at = next_poll
        subscription.poll_lock_owner = None
        subscription.poll_lock_expires_at = None
        await subscription.save(
            update_fields=[
                "poll_status",
                "poll_error_json",
                "poll_backoff_seconds",
                "next_poll_at",
                "poll_lock_owner",
                "poll_lock_expires_at",
            ]
        )

    async def _mark_error(
        self,
        subscription: TriggerSubscription,
        *,
        reason: str,
        detail: Optional[dict],
        delay_seconds: int,
    ) -> None:
        subscription.poll_status = "error"
        subscription.poll_error_json = {"reason": reason, "detail": detail}
        subscription.next_poll_at = _utcnow() + timedelta(seconds=delay_seconds)
        subscription.poll_lock_owner = None
        subscription.poll_lock_expires_at = None
        await subscription.save(
            update_fields=[
                "poll_status",
                "poll_error_json",
                "next_poll_at",
                "poll_lock_owner",
                "poll_lock_expires_at",
            ]
        )

    async def _disable_subscription(
        self,
        subscription: TriggerSubscription,
        *,
        reason: str,
        detail: Optional[dict] = None,
    ) -> None:
        subscription.poll_status = "disabled"
        subscription.poll_error_json = {"reason": reason, "detail": detail}
        subscription.poll_lock_owner = None
        subscription.poll_lock_expires_at = None
        await subscription.save(
            update_fields=[
                "poll_status",
                "poll_error_json",
                "poll_lock_owner",
                "poll_lock_expires_at",
            ]
        )

