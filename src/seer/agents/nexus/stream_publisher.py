"""
Redis Streams publisher for nexus agent SSE streaming.

Each agent execution session gets its own Redis Stream key.
Events are appended via XADD and consumed by the SSE endpoint via XREAD.

Redis Streams are chosen over Pub/Sub because:
- Persistent ordered log — clients can replay from any message ID
- SSE's Last-Event-ID header maps directly to Redis Stream message IDs
- Auto-expiry via TTL prevents unbounded key growth
"""
from typing import Optional

from seer.api.agents.workflow.chat_schema import StreamEvent, StreamEventType
from seer.logger import get_logger

logger = get_logger(__name__)

STREAM_TTL_SECONDS = 7200  # 2 hours
STREAM_KEY_PREFIX = "nexus:events"


class StreamPublisher:
    """
    Publishes agent execution events to a Redis Stream.

    Usage:
        publisher = StreamPublisher(session_id=42)
        await publisher.publish(StreamEventType.AGENT_START, {})
        ...
        await publisher.close()  # publishes DONE sentinel
    """

    def __init__(self, session_id: int):
        self.session_id = session_id
        self.stream_key = f"{STREAM_KEY_PREFIX}:{session_id}"
        self._redis: Optional[object] = None  # redis.asyncio.Redis, lazy init

    async def _get_redis(self):
        """Lazily create Redis connection."""
        if self._redis is None:
            import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel # Reason: Optional lazy import, redis may not be installed in all environments
            from seer.config import config  # pylint: disable=import-outside-toplevel # Reason: Avoids circular imports at module load time
            self._redis = aioredis.from_url(
                config.redis_url,
                decode_responses=True,
            )
        return self._redis

    async def publish(self, event_type: StreamEventType, data: dict) -> Optional[str]:
        """
        XADD event to stream and refresh TTL.

        Returns:
            Redis Stream message ID (e.g., '1709550000000-0'), or None on error
        """
        event = StreamEvent(type=event_type, data=data, session_id=self.session_id)
        try:
            r = await self._get_redis()
            msg_id = await r.xadd(self.stream_key, {"data": event.model_dump_json()})
            await r.expire(self.stream_key, STREAM_TTL_SECONDS)
            return msg_id
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Publisher must never crash the agent — streaming is best-effort
            logger.warning("StreamPublisher.publish failed for session=%d event=%s: %s", self.session_id, event_type.value, e)
            return None

    async def close(self) -> None:
        """Publish DONE sentinel and close Redis connection."""
        await self.publish(StreamEventType.DONE, {})
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:  # pylint: disable=broad-exception-caught # Reason: Close is best-effort
                pass
            self._redis = None

    async def publish_done(self) -> None:
        """Publish DONE sentinel without closing Redis connection (for reuse patterns)."""
        await self.publish(StreamEventType.DONE, {})
