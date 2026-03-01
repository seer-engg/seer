"""
Airtable "New Record in View" polling adapter.

Polls an Airtable view for new records using record ID tracking strategy.
This captures both newly created records AND existing records that move into
the view (by matching filter criteria).

Why Record ID Tracking vs Timestamps:
- Records can "move into" a view when their fields change to match view filters
- `createdTime` won't capture records that already existed but now match the view
- Record IDs are stable and unique (format: recXXXXXXXXXXXXXX)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

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

# Airtable API base URL
AIRTABLE_API_BASE = "https://api.airtable.com/v0"

# Default limits
DEFAULT_MAX_RECORDS = 100
MAX_SEEN_IDS_LIMIT = 10000  # Cap seen IDs to prevent unbounded growth


def _parse_created_time(value: Optional[str]) -> Optional[datetime]:
    """Parse Airtable createdTime string to datetime."""
    if not value:
        return None
    try:
        # Airtable format: "2026-02-27T10:00:00.000Z"
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class AirtableNewRecordInViewAdapter(PollAdapter):
    """
    Poll an Airtable view for new records.

    Cursor strategy: Track seen record IDs rather than timestamps.
    This captures both newly created records AND records that move into the view.
    """

    trigger_key = "poll.airtable.new_record_in_view"

    async def bootstrap_cursor(self, ctx: PollContext) -> Dict[str, Any]:
        """
        Initialize cursor with current view state (don't emit events for existing records).

        On first poll, we capture all existing record IDs so subsequent polls
        only trigger for truly "new" records in the view.
        """
        current_ids = await self._fetch_view_record_ids(ctx)
        return {
            "seen_record_ids": list(current_ids),
            "last_poll_time": datetime.now(timezone.utc).isoformat(),
        }

    async def poll(self, ctx: PollContext, cursor: Dict[str, Any]) -> PollResult:
        """
        Poll the view and return events for newly seen records.

        Compares current view record IDs against previously seen IDs.
        Records in current but not in seen are considered "new".
        """
        seen_ids: Set[str] = set(cursor.get("seen_record_ids", []))

        # Fetch current view records
        records = await self._fetch_view_records(ctx)
        current_ids: Set[str] = {r["id"] for r in records}

        # Find new records (in current but not in seen)
        new_ids = current_ids - seen_ids

        # Build events for new records
        config = ctx.subscription.provider_config or {}
        events: List[PolledEvent] = []

        for record in records:
            if record["id"] in new_ids:
                events.append(
                    PolledEvent(
                        payload=self._normalize_record(record, config),
                        raw=record,
                        provider_event_id=record["id"],
                        occurred_at=_parse_created_time(record.get("createdTime")),
                    )
                )

        # Update seen_ids: union of previous and current
        # Limit size to prevent unbounded growth for large/changing views
        updated_seen_ids = list((seen_ids | current_ids))
        if len(updated_seen_ids) > MAX_SEEN_IDS_LIMIT:
            # Keep only current IDs when limit exceeded (fresh start)
            updated_seen_ids = list(current_ids)
            logger.warning(
                "Airtable seen_record_ids exceeded %d limit, reset to current view state",
                MAX_SEEN_IDS_LIMIT,
            )

        return PollResult(
            events=events,
            cursor={
                "seen_record_ids": updated_seen_ids,
                "last_poll_time": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _fetch_view_records(self, ctx: PollContext) -> List[Dict[str, Any]]:
        """
        Fetch records from the configured Airtable view.

        Returns list of raw Airtable record objects.
        """
        config = ctx.subscription.provider_config or {}
        base_id = config.get("base_id")
        table_id = config.get("table_id")
        view_id = config.get("view_id")
        max_records = config.get("max_records", DEFAULT_MAX_RECORDS)

        if not all([base_id, table_id, view_id]):
            raise PollAdapterError(
                "Missing required config: base_id, table_id, view_id",
                permanent=True,
                detail={"config": config},
            )

        if not ctx.access_token:
            raise PollAdapterError(
                "Airtable access token required",
                permanent=True,
            )

        url = f"{AIRTABLE_API_BASE}/{base_id}/{table_id}"
        params: Dict[str, Any] = {
            "view": view_id,
            "maxRecords": min(max_records, DEFAULT_MAX_RECORDS),
        }
        headers = {
            "Authorization": f"Bearer {ctx.access_token}",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                await self._raise_for_status(response)
                data = response.json()
                return data.get("records", [])
        except PollAdapterError:
            raise
        except Exception as exc:
            logger.exception("Unexpected Airtable polling failure")
            raise PollAdapterError(
                "Unexpected Airtable polling failure",
                detail={"error": str(exc)},
            ) from exc

    async def _fetch_view_record_ids(self, ctx: PollContext) -> Set[str]:
        """Fetch just the record IDs from the view (used for bootstrap)."""
        records = await self._fetch_view_records(ctx)
        return {r["id"] for r in records}

    def _normalize_record(
        self, record: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalize Airtable record for event payload.

        Flattens fields to top level and adds metadata.
        """
        return {
            "record_id": record.get("id"),
            "created_time": record.get("createdTime"),
            "fields": record.get("fields", {}),
            "base_id": config.get("base_id"),
            "table_id": config.get("table_id"),
            "view_id": config.get("view_id"),
        }

    async def _raise_for_status(self, response: httpx.Response) -> None:
        """Handle Airtable API error responses."""
        if response.status_code < 400:
            return

        detail = {"status": response.status_code, "body": response.text[:500]}

        if response.status_code in {401, 403}:
            raise PollAdapterError(
                "Airtable authentication error",
                permanent=True,
                detail=detail,
            )

        if response.status_code == 429:
            raise PollAdapterError(
                "Airtable rate limited",
                backoff_seconds=60,
                detail=detail,
            )

        if response.status_code == 404:
            raise PollAdapterError(
                "Airtable base, table, or view not found",
                permanent=True,
                detail=detail,
            )

        raise PollAdapterError(
            f"Airtable API error: {response.status_code}",
            detail=detail,
        )


register_adapter(AirtableNewRecordInViewAdapter())
