"""
Meetings service — proxies Attendee API with org-level filtering.

Source of truth is the external Attendee service. No local DB models.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.config import config
from seer.logger import get_logger

logger = get_logger("services.meetings")

_REQUEST_TIMEOUT_SECONDS = 30


async def _attendee_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Make an authenticated request to the Attendee API."""
    api_key = config.meetings_ea_api_key
    base_url = config.meetings_ea_base_url
    if not api_key or not base_url:
        raise HTTPException(status_code=503, detail="meetingsEA is not configured")

    url = f"{base_url.rstrip('/')}{path}"  # pylint: disable=no-member  # Reason: Pydantic Field resolves to str at runtime
    headers = {"Authorization": f"Token {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.request(method, url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Meeting not found") from e
        logger.warning("Attendee API error: %s %s -> %d", method, path, e.response.status_code)
        raise HTTPException(status_code=502, detail="meetingsEA service error") from e
    except httpx.RequestError as e:
        logger.exception("Attendee API request failed: %s %s", method, path)
        raise HTTPException(status_code=502, detail="meetingsEA service unavailable") from e


def _matches_org(bot: Dict[str, Any], org_id: int) -> bool:
    """Check if a bot belongs to the given org."""
    metadata = bot.get("metadata") or {}
    return str(metadata.get("seer_org_id", "")) == str(org_id)


def _extract_created_at(bot: Dict[str, Any]) -> Optional[str]:
    """Extract created_at from the first event's timestamp."""
    events = bot.get("events") or []
    if events:
        return events[0].get("created_at")
    return bot.get("join_at")


def _normalize_bot(bot: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an Attendee bot response to our API format."""
    return {
        "bot_id": bot.get("id", ""),
        "state": bot.get("state", ""),
        "meeting_url": bot.get("meeting_url"),
        "join_at": bot.get("join_at"),
        "transcription_state": bot.get("transcription_state"),
        "recording_state": bot.get("recording_state"),
        "created_at": _extract_created_at(bot),
    }


async def list_meetings(org_id: int) -> List[Dict[str, Any]]:
    """List all bots for the given org, most recent first."""
    result = await _attendee_request("GET", "/api/v1/bots")
    bots_raw = result if isinstance(result, list) else result.get("results", result.get("bots", []))

    meetings = []
    for bot in bots_raw:
        if _matches_org(bot, org_id):
            meetings.append(_normalize_bot(bot))

    # Sort by created_at descending (newest first)
    meetings.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return meetings


async def get_meeting_detail(org_id: int, bot_id: str) -> Dict[str, Any]:
    """Get full detail for a single bot, with org verification."""
    bot = await _attendee_request("GET", f"/api/v1/bots/{bot_id}")
    if not _matches_org(bot, org_id):
        raise HTTPException(status_code=404, detail="Meeting not found")

    detail = _normalize_bot(bot)
    detail["events"] = bot.get("events", [])
    detail["metadata"] = bot.get("metadata", {})
    return detail


async def get_meeting_transcript(org_id: int, bot_id: str) -> List[Dict[str, Any]]:
    """Get normalized transcript for a bot, with org verification."""
    # Verify org ownership first
    bot = await _attendee_request("GET", f"/api/v1/bots/{bot_id}")
    if not _matches_org(bot, org_id):
        raise HTTPException(status_code=404, detail="Meeting not found")

    raw = await _attendee_request("GET", f"/api/v1/bots/{bot_id}/transcript")
    raw_utterances = raw if isinstance(raw, list) else raw.get("utterances", [])

    utterances = []
    for u in raw_utterances:
        transcription = u.get("transcription") or {}
        text = transcription.get("transcript", "") if isinstance(transcription, dict) else str(transcription)
        if not text:
            continue
        utterances.append({
            "speaker": u.get("speaker_name", "Unknown"),
            "text": text,
            "start_ms": u.get("timestamp_ms", 0),
            "duration_ms": u.get("duration_ms", 0),
        })

    return utterances


async def get_meeting_recording_url(org_id: int, bot_id: str) -> Dict[str, Any]:
    """Get recording URL for a bot, with org verification."""
    # Verify org ownership first
    bot = await _attendee_request("GET", f"/api/v1/bots/{bot_id}")
    if not _matches_org(bot, org_id):
        raise HTTPException(status_code=404, detail="Meeting not found")

    try:
        recording = await _attendee_request("GET", f"/api/v1/bots/{bot_id}/recording")
        return {
            "url": recording.get("url"),
            "start_timestamp_ms": recording.get("start_timestamp_ms"),
        }
    except HTTPException as e:
        if e.status_code == 404:
            return {"url": None, "start_timestamp_ms": None}
        raise
