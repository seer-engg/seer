"""
Server-Sent Events (SSE) async generator for nexus agent streaming.

Reads events from a Redis Stream and yields SSE-formatted strings.
Reconnect support: client sends Last-Event-ID header → resumes from that message.

SSE wire format per event:
    id: {redis_msg_id}\\ndata: {json}\\n\\n
"""
import asyncio
import json
from typing import AsyncIterator, Optional

from seer.api.agents.workflow.chat_schema import StreamEvent, StreamEventType
from seer.logger import get_logger

logger = get_logger(__name__)

STREAM_KEY_PREFIX = "nexus:events"
XREAD_BLOCK_MS = 5000       # Block up to 5s waiting for new messages
MAX_MESSAGES_PER_READ = 100  # Batch size for XREAD
DB_FALLBACK_BATCH = 50       # DB messages to load for history fallback


def _build_stream_key(session_id: int) -> str:
    return f"{STREAM_KEY_PREFIX}:{session_id}"


async def get_stream_watermark(session_id: int) -> str:
    """
    Return the ID of the last message currently in the Redis stream for a session.

    Use this before publishing new events so you can pass the result as
    `last_event_id` to `stream_events_sse()`.  The SSE stream will then start
    *after* this ID, skipping all historical events and delivering only the
    newly-published ones.

    Returns "0" when the stream doesn't exist yet (the SSE stream will block
    until the first event appears).
    """
    import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel # Reason: Lazy import mirrors the rest of this module
    from seer.config import config  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import

    stream_key = _build_stream_key(session_id)
    r = aioredis.from_url(config.redis_url, decode_responses=True)
    try:
        last_messages = await r.xrevrange(stream_key, count=1)
        return last_messages[0][0] if last_messages else "0"
    finally:
        try:
            await r.aclose()
        except Exception:  # pylint: disable=broad-exception-caught # Reason: Close is best-effort
            pass


def _format_sse(msg_id: str, data: str) -> str:
    """Format a single SSE message."""
    return f"id: {msg_id}\ndata: {data}\n\n"


async def stream_events_sse(
    session_id: int,
    last_event_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Read events from Redis Stream and yield SSE-formatted strings.

    Args:
        session_id: Chat session ID — determines the Redis stream key
        last_event_id: Redis message ID to resume from (from SSE Last-Event-ID header).
                       None means start from beginning of stream (full replay).

    Yields:
        SSE-formatted strings ready to be written to the HTTP response
    """
    import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel # Reason: Optional lazy import
    from seer.config import config  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import at module load time

    stream_key = _build_stream_key(session_id)
    # Redis cursor: '0' replays from beginning; specific ID resumes from after that message
    cursor = last_event_id if last_event_id else "0"

    r = aioredis.from_url(config.redis_url, decode_responses=True)
    try:
        # Check if stream exists
        stream_exists = await r.exists(stream_key)

        if not stream_exists:
            # Stream key missing — either TTL expired or agent hasn't started yet
            async for event_str in _handle_missing_stream(r, session_id, stream_key, cursor):
                yield event_str
            return

        # Main loop: read from stream until DONE event
        async for event_str in _read_stream_loop(r, stream_key, cursor):
            yield event_str

    finally:
        try:
            await r.aclose()
        except Exception:  # pylint: disable=broad-exception-caught # Reason: Close is best-effort
            pass


async def _read_stream_loop(r, stream_key: str, cursor: str) -> AsyncIterator[str]:
    # pylint: disable=too-complex # Reason: Multiple Redis polling states and error conditions handled in single loop
    """Read events from Redis Stream, blocking between reads until DONE."""
    while True:
        try:
            # XREAD BLOCK: blocks up to XREAD_BLOCK_MS ms, returns list of (stream_key, [(msg_id, fields), ...])
            results = await r.xread(
                {stream_key: cursor},
                count=MAX_MESSAGES_PER_READ,
                block=XREAD_BLOCK_MS,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Redis errors should not crash the SSE generator
            logger.warning("SSE xread error for stream %s: %s", stream_key, e)
            await asyncio.sleep(1)
            continue

        if not results:
            # Timeout with no new messages — check if stream still exists (TTL expired mid-run)
            exists = await r.exists(stream_key)
            if not exists:
                logger.info("SSE stream key expired mid-run: %s", stream_key)
                return
            continue

        # results: [(stream_key_bytes, [(msg_id, {field: value}), ...])]
        for _stream_name, messages in results:
            for msg_id, fields in messages:
                cursor = msg_id
                raw = fields.get("data", "{}")
                yield _format_sse(msg_id, raw)

                # Check if this is the DONE sentinel
                try:
                    event_dict = json.loads(raw)
                    if event_dict.get("type") == StreamEventType.DONE.value:
                        return
                except (json.JSONDecodeError, AttributeError):
                    pass


async def _handle_missing_stream(r, session_id: int, stream_key: str, cursor: str) -> AsyncIterator[str]:
    # pylint: disable=too-complex,too-many-locals # Reason: Handles multiple session status/fallback paths with distinct local vars per case
    """
    Handle the case where the Redis stream key doesn't exist.

    Cases:
    1. Stream TTL expired and session is COMPLETED → fallback to DB history
    2. Agent hasn't started yet (QUEUED) → wait with back-off for key to appear
    3. Session FAILED → emit error + done
    """
    from seer.database.workflow_models import WorkflowChatSession, ChatExecutionStatus  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import
    from seer.database import WorkflowChatMessage  # pylint: disable=import-outside-toplevel # Reason: Avoids circular import

    try:
        session = await WorkflowChatSession.get_or_none(id=session_id)
        if session is None:
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={"message": "Session not found", "status_code": 404},
                session_id=session_id,
            )
            yield _format_sse("0-0", error_event.model_dump_json())
            done_event = StreamEvent(type=StreamEventType.DONE, data={}, session_id=session_id)
            yield _format_sse("0-1", done_event.model_dump_json())
            return

        status = session.current_execution_status

        if status == ChatExecutionStatus.COMPLETED:
            # TTL expired — load history from DB and emit as message_history events
            messages = await WorkflowChatMessage.filter(
                session_id=session_id
            ).order_by("created_at").limit(DB_FALLBACK_BATCH).all()

            for i, msg in enumerate(messages):
                history_event = StreamEvent(
                    type=StreamEventType.AGENT_END if msg.role == "assistant" else StreamEventType.AI_MESSAGE,
                    data={"content": msg.content, "role": msg.role},
                    session_id=session_id,
                )
                yield _format_sse(f"db-{i}", history_event.model_dump_json())

            done_event = StreamEvent(type=StreamEventType.DONE, data={}, session_id=session_id)
            yield _format_sse("db-done", done_event.model_dump_json())
            return

        if status == ChatExecutionStatus.FAILED:
            error_detail = session.current_execution_error or {}
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                data={"message": error_detail.get("detail", "Execution failed"), "status_code": error_detail.get("status", 500)},
                session_id=session_id,
            )
            yield _format_sse("0-0", error_event.model_dump_json())
            done_event = StreamEvent(type=StreamEventType.DONE, data={}, session_id=session_id)
            yield _format_sse("0-1", done_event.model_dump_json())
            return

        if status == ChatExecutionStatus.INTERRUPTED:
            interrupt_data = session.pending_interrupt_data or {}
            interrupt_event = StreamEvent(
                type=StreamEventType.INTERRUPT,
                data=interrupt_data,
                session_id=session_id,
            )
            yield _format_sse("0-0", interrupt_event.model_dump_json())
            done_event = StreamEvent(type=StreamEventType.DONE, data={}, session_id=session_id)
            yield _format_sse("0-1", done_event.model_dump_json())
            return

        # QUEUED or RUNNING — agent hasn't started publishing yet, wait with back-off
        backoff_seconds = [0.5, 1, 1, 2, 2, 3, 3, 5, 5, 5]
        for wait in backoff_seconds:
            await asyncio.sleep(wait)
            exists = await r.exists(stream_key)
            if exists:
                # Stream appeared — hand off to main loop
                async for event_str in _read_stream_loop(r, stream_key, cursor):
                    yield event_str
                return

        # Stream never appeared — emit timeout error
        logger.warning("SSE stream never appeared after back-off for session=%d", session_id)
        error_event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"message": "Agent did not start within expected time", "status_code": 504},
            session_id=session_id,
        )
        yield _format_sse("0-0", error_event.model_dump_json())
        done_event = StreamEvent(type=StreamEventType.DONE, data={}, session_id=session_id)
        yield _format_sse("0-1", done_event.model_dump_json())

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: SSE generator must always yield DONE to terminate client
        logger.error("SSE _handle_missing_stream error session=%d: %s", session_id, e)
        error_event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"message": "Internal error", "status_code": 500},
            session_id=session_id,
        )
        yield _format_sse("0-0", error_event.model_dump_json())
        done_event = StreamEvent(type=StreamEventType.DONE, data={}, session_id=session_id)
        yield _format_sse("0-1", done_event.model_dump_json())
