from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import AsyncIterator, Optional

from seer.logger import get_logger
from seer.services.collaboration.models import CollaborationEventType
from seer.services.collaboration.publisher import ORG_STREAM_KEY_PREFIX

logger = get_logger(__name__)

XREAD_BLOCK_MS = 1000
HEARTBEAT_INTERVAL_SECONDS = 25
MAX_MESSAGES_PER_READ = 100


def _build_stream_key(organization_id: int) -> str:
    return f"{ORG_STREAM_KEY_PREFIX}:{organization_id}"


def _format_sse(msg_id: str, data: str) -> str:
    return f"id: {msg_id}\nevent: collaboration\ndata: {data}\n\n"


async def _get_initial_cursor_and_sync_event(
    redis_client: object,
    stream_key: str,
    last_event_id: str | None,
    organization_id: int,
) -> tuple[str, str | None]:
    cursor = last_event_id if last_event_id else "$"
    if not last_event_id or await redis_client.exists(stream_key):
        return cursor, None

    return (
        "$",
        _format_sse(
            "sync-required",
            (
                '{"event_type":"%s","organization_id":%d,"resource_type":"organization","payload":{"reason":"stream_missing"}}'
                % (CollaborationEventType.SYNC_REQUIRED.value, organization_id)
            ),
        ),
    )


async def _read_stream(redis_client: object, stream_key: str, cursor: str) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
    return await redis_client.xread(
        {stream_key: cursor},
        count=MAX_MESSAGES_PER_READ,
        block=XREAD_BLOCK_MS,
    )


def _yield_messages(
    results: list[tuple[str, list[tuple[str, dict[str, str]]]]],
    cursor: str,
) -> tuple[str, list[str]]:
    chunks: list[str] = []
    next_cursor = cursor
    for _stream_name, messages in results:
        for msg_id, fields in messages:
            next_cursor = msg_id
            chunks.append(_format_sse(msg_id, fields.get("data", "{}")))
    return next_cursor, chunks


async def _read_next_chunks(
    redis_client: object,
    stream_key: str,
    cursor: str,
    organization_id: int,
    last_heartbeat_at: float,
) -> tuple[str, float, list[str]]:
    try:
        results = await _read_stream(redis_client, stream_key, cursor)
    except Exception as exc:  # pylint: disable=broad-exception-caught # Reason: SSE should survive transient Redis issues
        logger.warning("Collaboration SSE xread error for org=%s: %s", organization_id, exc)
        await asyncio.sleep(1)
        return cursor, last_heartbeat_at, []

    if results:
        next_cursor, chunks = _yield_messages(results, cursor)
        return next_cursor, last_heartbeat_at, chunks

    now = asyncio.get_running_loop().time()
    if now - last_heartbeat_at >= HEARTBEAT_INTERVAL_SECONDS:
        return cursor, now, [": heartbeat\n\n"]
    return cursor, last_heartbeat_at, []


async def stream_org_events_sse(
    organization_id: int,
    last_event_id: Optional[str] = None,
    should_stop: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel # Reason: optional lazy import
    from seer.config import config  # pylint: disable=import-outside-toplevel # Reason: avoid circular import at module load time

    stream_key = _build_stream_key(organization_id)
    redis = aioredis.from_url(config.redis_url, decode_responses=True)
    last_heartbeat_at = asyncio.get_running_loop().time() - HEARTBEAT_INTERVAL_SECONDS

    try:
        cursor, sync_event = await _get_initial_cursor_and_sync_event(redis, stream_key, last_event_id, organization_id)
        if sync_event is not None:
            yield sync_event

        while True:
            if should_stop is not None and await should_stop():
                return

            cursor, last_heartbeat_at, chunks = await _read_next_chunks(
                redis,
                stream_key,
                cursor,
                organization_id,
                last_heartbeat_at,
            )
            if should_stop is not None and await should_stop():
                return
            for chunk in chunks:
                yield chunk
    finally:
        try:
            await redis.aclose()
        except Exception:  # pylint: disable=broad-exception-caught # Reason: close is best-effort
            pass
