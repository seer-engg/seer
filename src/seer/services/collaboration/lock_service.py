from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from seer.database import User
from seer.logger import get_logger

from .models import WorkflowLockMetadata

logger = get_logger(__name__)

LOCK_TTL_SECONDS = 30


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _expires_at() -> datetime:
    return _now_utc() + timedelta(seconds=LOCK_TTL_SECONDS)


def _holder_name(user: User) -> str:
    display_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return display_name or user.email or user.user_id


class WorkflowLockService:
    """Redis-backed workflow lease locks for org collaboration."""

    def __init__(self):
        self._redis: Optional[object] = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel # Reason: optional lazy import
            from seer.config import config  # pylint: disable=import-outside-toplevel # Reason: avoid circular import at module load time

            self._redis = aioredis.from_url(
                config.redis_url,
                decode_responses=True,
            )
        return self._redis

    @staticmethod
    def build_lock_key(organization_id: int, workflow_id: str) -> str:
        return f"org:{organization_id}:workflow:{workflow_id}:lock"

    async def close(self) -> None:
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:  # pylint: disable=broad-exception-caught # Reason: close is best-effort
                pass
            self._redis = None

    async def get_lock(self, organization_id: int, workflow_id: str) -> WorkflowLockMetadata | None:
        redis = await self._get_redis()
        raw_value = await redis.get(self.build_lock_key(organization_id, workflow_id))
        return self._decode_lock(raw_value, organization_id=organization_id, workflow_id=workflow_id)

    async def acquire_lock(
        self,
        *,
        organization_id: int,
        workflow_id: str,
        user: User,
        tab_id: str | None = None,
    ) -> tuple[bool, WorkflowLockMetadata]:
        redis = await self._get_redis()
        lock = self._build_lock(organization_id=organization_id, workflow_id=workflow_id, user=user, tab_id=tab_id)
        key = self.build_lock_key(organization_id, workflow_id)

        acquired = await redis.set(key, lock.model_dump_json(), ex=LOCK_TTL_SECONDS, nx=True)
        if acquired:
            return True, lock

        existing = await self.get_lock(organization_id, workflow_id)
        if existing is not None:
            return False, existing

        # Retry once in case the key disappeared between SET NX and GET.
        acquired = await redis.set(key, lock.model_dump_json(), ex=LOCK_TTL_SECONDS, nx=True)
        if acquired:
            return True, lock

        existing = await self.get_lock(organization_id, workflow_id)
        return False, existing or lock

    async def heartbeat_lock(
        self,
        *,
        organization_id: int,
        workflow_id: str,
        user: User,
        tab_id: str | None = None,
    ) -> WorkflowLockMetadata | None:
        current_lock = await self.get_lock(organization_id, workflow_id)
        if current_lock is None or not self._is_holder(current_lock, user=user, tab_id=tab_id):
            return None

        refreshed_lock = self._build_lock(
            organization_id=organization_id,
            workflow_id=workflow_id,
            user=user,
            tab_id=current_lock.tab_id or tab_id,
            acquired_at=current_lock.acquired_at,
        )
        redis = await self._get_redis()
        await redis.set(
            self.build_lock_key(organization_id, workflow_id),
            refreshed_lock.model_dump_json(),
            ex=LOCK_TTL_SECONDS,
            xx=True,
        )
        return refreshed_lock

    async def release_lock(
        self,
        *,
        organization_id: int,
        workflow_id: str,
        user: User,
        tab_id: str | None = None,
    ) -> WorkflowLockMetadata | None:
        current_lock = await self.get_lock(organization_id, workflow_id)
        if current_lock is None or not self._is_holder(current_lock, user=user, tab_id=tab_id):
            return None

        redis = await self._get_redis()
        await redis.delete(self.build_lock_key(organization_id, workflow_id))
        return current_lock

    def _build_lock(
        self,
        *,
        organization_id: int,
        workflow_id: str,
        user: User,
        tab_id: str | None = None,
        acquired_at: datetime | None = None,
    ) -> WorkflowLockMetadata:
        return WorkflowLockMetadata(
            organization_id=organization_id,
            workflow_id=workflow_id,
            holder_clerk_user_id=user.user_id,
            holder_db_user_id=user.id,
            holder_name=_holder_name(user),
            acquired_at=acquired_at or _now_utc(),
            expires_at=_expires_at(),
            tab_id=tab_id,
        )

    @staticmethod
    def _decode_lock(
        raw_value: str | None,
        *,
        organization_id: int,
        workflow_id: str,
    ) -> WorkflowLockMetadata | None:
        if not raw_value:
            return None
        try:
            return WorkflowLockMetadata.model_validate_json(raw_value)
        except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: tolerate malformed Redis state and treat as missing
            logger.warning(
                "Invalid workflow lock payload for org=%s workflow=%s: %s",
                organization_id,
                workflow_id,
                exc,
            )
            return None

    @staticmethod
    def _is_holder(lock: WorkflowLockMetadata, *, user: User, tab_id: str | None = None) -> bool:
        if lock.holder_db_user_id != user.id and lock.holder_clerk_user_id != user.user_id:
            return False
        if tab_id and lock.tab_id and lock.tab_id != tab_id:
            return False
        return True
