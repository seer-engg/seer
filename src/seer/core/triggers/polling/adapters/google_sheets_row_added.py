"""
Google Sheets "Row Added" polling adapter.

Polls a Google Sheet for new rows appended at the bottom using row-count tracking.
On each poll, reads rows starting from last_row_count + 1 and emits new rows as events.

Why Row Count Tracking:
- Google Sheets rows have no stable unique IDs (unlike Airtable records)
- The Sheets API does not provide change tracking or webhooks for row additions
- Tracking the row count and reading from the last known position is simple and reliable
- This matches the approach used by n8n and similar platforms
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import zip_longest
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

SHEETS_API_BASE = "https://sheets.googleapis.com/v4/spreadsheets"
DEFAULT_MAX_NEW_ROWS = 100
DEFAULT_SHEET_NAME = "Sheet1"


class GoogleSheetsRowAddedAdapter(PollAdapter):
    """
    Poll a Google Sheet for newly appended rows.

    Cursor strategy: Track the total row count. On each poll, read rows starting
    from last_row_count + 1. Only rows added at the bottom are detected.
    """

    trigger_key = "poll.google_sheets.row_added"

    async def bootstrap_cursor(self, ctx: PollContext) -> Dict[str, Any]:
        """
        Initialize cursor with current row count so existing rows don't trigger events.

        Also captures header row (first row) if has_header_row is configured.
        """
        config = ctx.subscription.provider_config or {}
        spreadsheet_id = config.get("spreadsheet_id")
        sheet_name = config.get("sheet_name", DEFAULT_SHEET_NAME)
        has_header_row = config.get("has_header_row", True)

        if not spreadsheet_id:
            raise PollAdapterError(
                "Missing required config: spreadsheet_id",
                permanent=True,
                detail={"config": config},
            )

        rows = await self._fetch_rows(ctx, spreadsheet_id, sheet_name)
        row_count = len(rows)
        headers: Optional[List[str]] = None

        if has_header_row and row_count > 0:
            headers = [str(cell) for cell in rows[0]]

        return {
            "last_row_count": row_count,
            "headers": headers,
            "last_poll_time": datetime.now(timezone.utc).isoformat(),
        }

    async def poll(self, ctx: PollContext, cursor: Dict[str, Any]) -> PollResult:
        """
        Poll for new rows appended after the last known row.

        Reads from row last_row_count+1 onward, emits each new row as an event.
        """
        config = ctx.subscription.provider_config or {}
        spreadsheet_id = config.get("spreadsheet_id")
        sheet_name = config.get("sheet_name", DEFAULT_SHEET_NAME)

        if not spreadsheet_id:
            raise PollAdapterError(
                "Missing required config: spreadsheet_id",
                permanent=True,
                detail={"config": config},
            )

        if not ctx.access_token:
            raise PollAdapterError(
                "Google Sheets access token required",
                permanent=True,
            )

        last_row_count = int(cursor.get("last_row_count", 0))
        headers: Optional[List[str]] = cursor.get("headers")
        start_row = last_row_count + 1
        end_row = start_row + DEFAULT_MAX_NEW_ROWS - 1

        # Read only the new rows range
        range_str = f"{sheet_name}!A{start_row}:ZZZ{end_row}"
        new_rows = await self._fetch_rows(ctx, spreadsheet_id, range_str)

        if not new_rows:
            return PollResult(events=[], cursor=cursor)

        events: List[PolledEvent] = []
        for i, row in enumerate(new_rows):
            row_number = last_row_count + i + 1
            normalized = self._normalize_row(row, row_number, headers, config)
            events.append(
                PolledEvent(
                    payload=normalized,
                    raw={"row_number": row_number, "values": row},
                    provider_event_id=f"{spreadsheet_id}:{sheet_name}:{row_number}",
                    occurred_at=datetime.now(timezone.utc),
                )
            )

        new_row_count = last_row_count + len(new_rows)
        has_more = len(new_rows) >= DEFAULT_MAX_NEW_ROWS

        return PollResult(
            events=events,
            cursor={
                "last_row_count": new_row_count,
                "headers": headers,
                "last_poll_time": datetime.now(timezone.utc).isoformat(),
            },
            has_more=has_more,
        )

    async def _fetch_rows(
        self, ctx: PollContext, spreadsheet_id: str, range_str: str
    ) -> List[List[Any]]:
        """Fetch rows from the Google Sheets API."""
        if not ctx.access_token:
            raise PollAdapterError(
                "Google Sheets access token required",
                permanent=True,
            )

        url = f"{SHEETS_API_BASE}/{spreadsheet_id}/values/{range_str}"
        headers = {
            "Authorization": f"Bearer {ctx.access_token}",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                await self._raise_for_status(response)
                data = response.json()
                return data.get("values", [])
        except PollAdapterError:
            raise
        except Exception as exc:
            logger.exception("Unexpected Google Sheets polling failure")
            raise PollAdapterError(
                "Unexpected Google Sheets polling failure",
                detail={"error": str(exc)},
            ) from exc

    def _normalize_row(
        self,
        row: List[Any],
        row_number: int,
        headers: Optional[List[str]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize a sheet row into an event payload."""
        spreadsheet_id = config.get("spreadsheet_id", "")
        sheet_name = config.get("sheet_name", DEFAULT_SHEET_NAME)

        base: Dict[str, Any] = {
            "row_number": row_number,
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,
        }

        if headers:
            fields = dict(zip_longest(headers, row, fillvalue=None))
            base["fields"] = fields
            base["row_values"] = None
        else:
            base["fields"] = None
            base["row_values"] = row

        return base

    async def _raise_for_status(self, response: httpx.Response) -> None:
        """Handle Google Sheets API error responses."""
        if response.status_code < 400:
            return

        detail = {"status": response.status_code, "body": response.text[:500]}

        if response.status_code in {401, 403}:
            raise PollAdapterError(
                "Google Sheets authentication error",
                permanent=True,
                detail=detail,
            )

        if response.status_code == 429:
            raise PollAdapterError(
                "Google Sheets rate limited",
                backoff_seconds=60,
                detail=detail,
            )

        if response.status_code == 404:
            raise PollAdapterError(
                "Spreadsheet or sheet not found",
                permanent=True,
                detail=detail,
            )

        raise PollAdapterError(
            f"Google Sheets API error: {response.status_code}",
            detail=detail,
        )


register_adapter(GoogleSheetsRowAddedAdapter())
