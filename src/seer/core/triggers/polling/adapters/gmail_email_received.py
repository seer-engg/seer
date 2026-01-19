from __future__ import annotations

import email.utils
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

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

GMAIL_API_BASE = "https://www.googleapis.com/gmail/v1/users/me"
DEFAULT_OVERLAP_MS = 5 * 60 * 1000  # 5 minutes
MAX_MESSAGES_PER_POLL = 25


def _utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _parse_address(value: Optional[str]) -> Dict[str, Optional[str]]:
    name, addr = email.utils.parseaddr(value or "")
    return {"name": name or None, "email": addr or None}


def _parse_address_list(value: Optional[str]) -> List[Dict[str, Optional[str]]]:
    if not value:
        return []
    # email.utils.getaddresses handles comma-separated lists
    entries = email.utils.getaddresses([value])
    return [{"name": name or None, "email": addr or None} for name, addr in entries]


class GmailEmailReceivedAdapter(PollAdapter):
    """Poll Gmail inbox for new messages using watermark + overlap strategy."""

    trigger_key = "poll.gmail.email_received"

    async def bootstrap_cursor(self, ctx: PollContext) -> Dict[str, Any]:
        now_ms = _utc_ms(datetime.now(timezone.utc))
        overlap_ms = ctx.subscription.provider_config.get("overlap_ms", DEFAULT_OVERLAP_MS) if ctx.subscription.provider_config else DEFAULT_OVERLAP_MS
        return {"watermark_ms": now_ms, "overlap_ms": overlap_ms}

    async def poll(self, ctx: PollContext, cursor: Dict[str, Any]) -> PollResult:
        label_ids = self._resolve_label_ids(ctx)
        max_results = self._resolve_max_results(ctx)
        overlap_ms = int(cursor.get("overlap_ms") or DEFAULT_OVERLAP_MS)
        watermark_ms = int(cursor.get("watermark_ms") or _utc_ms(datetime.now(timezone.utc)))
        after_ms = max(0, watermark_ms - overlap_ms)
        after_seconds = after_ms // 1000
        query_parts = [ctx.subscription.provider_config.get("query")] if ctx.subscription.provider_config else []
        query_parts = [part for part in query_parts if part]
        query_parts.append(f"after:{after_seconds}")
        query = " ".join(query_parts).strip()

        headers = {"Authorization": f"Bearer {ctx.access_token}", "Accept": "application/json"}
        params: Dict[str, Any] = {"maxResults": max_results}
        if label_ids:
            params["labelIds"] = ",".join(label_ids)
        if query:
            params["q"] = query

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                list_resp = await client.get(f"{GMAIL_API_BASE}/messages", headers=headers, params=params)
                await self._raise_for_status(list_resp)
                list_data = list_resp.json()
                messages = list_data.get("messages", []) or []
                if not messages:
                    return PollResult(events=[], cursor={"watermark_ms": watermark_ms, "overlap_ms": overlap_ms})

                polled_events: List[PolledEvent] = []
                new_watermark = watermark_ms
                for message in messages[:max_results]:
                    msg_id = message.get("id")
                    if not msg_id:
                        continue
                    # Fetch metadata for each message
                    msg_resp = await client.get(
                        f"{GMAIL_API_BASE}/messages/{msg_id}",
                        headers=headers,
                        params={"format": "metadata", "metadataHeaders": "From,To,Subject,Date"},
                    )
                    await self._raise_for_status(msg_resp)
                    msg_data = msg_resp.json()
                    internal_date_ms = int(msg_data.get("internalDate") or 0)
                    new_watermark = max(new_watermark, internal_date_ms)
                    normalized_payload = self._normalize_message(msg_data)
                    polled_events.append(
                        PolledEvent(
                            payload=normalized_payload,
                            raw=msg_data,
                            provider_event_id=msg_data.get("id"),
                            occurred_at=datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc),
                        )
                    )

                cursor_payload = {"watermark_ms": new_watermark, "overlap_ms": overlap_ms}
                has_more = bool(list_data.get("nextPageToken"))
                return PollResult(events=polled_events, cursor=cursor_payload, has_more=has_more)
        except PollAdapterError:
            raise
        except Exception as exc:
            logger.exception("Unexpected Gmail polling failure")
            raise PollAdapterError("Unexpected Gmail polling failure", detail={"error": str(exc)})

    def _normalize_message(self, msg_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = msg_data.get("payload") or {}
        headers = {item["name"]: item["value"] for item in (payload.get("headers") or []) if item.get("name") and item.get("value")}
        from_header = headers.get("From")
        to_header = headers.get("To")

        return {
            "message_id": msg_data.get("id"),
            "thread_id": msg_data.get("threadId"),
            "internal_date_ms": int(msg_data.get("internalDate") or 0),
            "snippet": msg_data.get("snippet"),
            "subject": headers.get("Subject"),
            "from": _parse_address(from_header),
            "to": _parse_address_list(to_header),
            "date_header": headers.get("Date"),
            "labels": msg_data.get("labelIds", []),
            "history_id": msg_data.get("historyId"),
        }

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = {"status": response.status_code, "body": response.text[:500]}
        if response.status_code in {401, 403}:
            raise PollAdapterError("Gmail authentication error", permanent=True, detail=detail)
        if response.status_code == 429:
            raise PollAdapterError("Gmail rate limited", backoff_seconds=60, detail=detail)
        raise PollAdapterError("Gmail API error", detail=detail)

    def _resolve_label_ids(self, ctx: PollContext) -> List[str]:
        config = ctx.subscription.provider_config or {}
        label_ids = config.get("label_ids")
        if isinstance(label_ids, list):
            return [str(item) for item in label_ids if str(item).strip()]
        return ["INBOX"]

    def _resolve_max_results(self, ctx: PollContext) -> int:
        config = ctx.subscription.provider_config or {}
        value = config.get("max_results", MAX_MESSAGES_PER_POLL)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = MAX_MESSAGES_PER_POLL
        return max(1, min(numeric, MAX_MESSAGES_PER_POLL))


register_adapter(GmailEmailReceivedAdapter())
