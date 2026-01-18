from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from typing import Callable
from uuid import uuid4

from fastapi import HTTPException
from tortoise.expressions import Q
from tortoise.transactions import in_transaction

from seer.core.triggers.polling.adapters.base import (
    PollAdapterError,
    PollContext,
    adapter_registry,
)
from seer.core.triggers.polling.dedupe import compute_event_hash
from seer.database import TriggerSubscription
from seer.logger import get_logger
from seer.tools.oauth_manager import get_oauth_token
from seer.core.registry.trigger_registry import trigger_registry
from seer.core.triggers.events import build_event_envelope,persist_event


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
        trigger_event_dispatcher: Callable,
    ) -> None:
        self.lock_timeout = timedelta(seconds=lock_timeout_seconds)
        self.max_batch_size = max_batch_size
        self.worker_id = f"poller-{uuid4().hex[:8]}"
        self.trigger_event_dispatcher = trigger_event_dispatcher

    async def tick(self) -> None:
        # logger.info("Polling Engine Tick")
        subscriptions = await self._lease_due_subscriptions(limit=self.max_batch_size)
        if not subscriptions:
            # logger.info("No subscriptions to process")
            return

        for subscription in subscriptions:
            logger.info("Processing subscription %s", subscription.id)
            try:
                await self._process_subscription(subscription)
            except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all subscription errors to mark as failed
                logger.exception(
                    "Failed to process trigger subscription",
                    extra={"subscription_id": subscription.id, "trigger_key": subscription.trigger_key},
                )

                # Track trigger poll error to PostHog
                await subscription.fetch_related("user")
                if subscription.user:
                    from seer.analytics import analytics  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
                    analytics.capture_error(
                        distinct_id=subscription.user.user_id,
                        error=e,
                        context={
                            "subscription_id": str(subscription.id),
                            "trigger_key": subscription.trigger_key,
                        },
                        error_location="trigger_poll",
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
                    is_polling=True,
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

    async def _get_oauth_connection(self, subscription: TriggerSubscription, user):
        """Get OAuth connection for subscription. Returns (connection, access_token) or None if error handled."""
        definition = trigger_registry.get(subscription.trigger_key)
        if subscription.provider_connection_id is None:
            if definition.meta.requires_connection:
                logger.error(
                    "Missing provider connection for subscription",
                    extra={"subscription_id": subscription.id, "trigger_key": subscription.trigger_key},
                )
                await self._mark_error(
                    subscription,
                    reason="missing_provider_connection",
                    detail={"trigger_key": subscription.trigger_key},
                    delay_seconds=max(subscription.poll_interval_seconds, 60),
                )
            return None

        try:
            return await get_oauth_token(
                user,
                connection_id=str(subscription.provider_connection_id),
            )
        except HTTPException as exc:
            should_disable = exc.status_code in {401, 403, 404}
            if should_disable:
                logger.error("OAuth error for subscription %s: %s", subscription.id, exc.detail)
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
            return None

    async def _process_subscription(self, subscription: TriggerSubscription) -> None:
        adapter = adapter_registry.get(subscription.trigger_key)
        if adapter is None:
            logger.error(
                "No poll adapter registered for trigger",
                extra={"subscription_id": subscription.id, "trigger_key": subscription.trigger_key},
            )
            await self._disable_subscription(subscription, reason="missing_adapter")
            return

        await subscription.fetch_related("user")
        user = subscription.user
        if user is None:
            logger.error("Missing user for subscription %s", subscription.id)
            await self._disable_subscription(subscription, reason="missing_user")
            return

        # Get OAuth connection if needed
        oauth_result = await self._get_oauth_connection(subscription, user)
        definition = trigger_registry.get(subscription.trigger_key)
        if oauth_result is None and definition.meta.requires_connection:
            return
        connection, access_token = oauth_result if oauth_result else (None, None)

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
            logger.error("Poll adapter error for subscription %s: %s", subscription.id, exc.detail)
            if exc.permanent:
                logger.error("Permanent poll adapter error for subscription %s", subscription.id)
                await self._disable_subscription(
                    subscription, reason="adapter_permanent_error", detail=exc.detail
                )
                return
            backoff = exc.backoff_seconds or min(subscription.poll_interval_seconds * 2, 600)
            logger.error("Backoff poll adapter error for subscription %s: %s", subscription.id, backoff)
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
            logger.info("No events to handle for subscription %s", subscription.id)
            return

        # Verify workflow exists before dispatching events
        await subscription.fetch_related("workflow")
        if not subscription.workflow:
            logger.error(
                "Workflow missing for subscription, disabling",
                extra={"subscription_id": subscription.id, "trigger_key": subscription.trigger_key},
            )
            await self._disable_subscription(subscription, reason="missing_workflow")
            return

        provider = trigger_registry.get(subscription.trigger_key).provider
        logger.info("Loading trigger provider for subscription %s", subscription.id)
        for polled in events:
            logger.info("Building event envelope for subscription %s", subscription.id)
            envelope = build_event_envelope(
                trigger_id=subscription.trigger_id,
                trigger_key=subscription.trigger_key,
                title=subscription.title,
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

            event, created = await persist_event(
                subscription=subscription,
                envelope=envelope,
                provider_event_id=provider_event_id,
                event_hash=event_hash,
                raw=polled.raw,
            )
            if created:
                logger.info("Dispatching trigger event for subscription %s", subscription.id)
                await self.trigger_event_dispatcher(subscription, event, envelope)

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
