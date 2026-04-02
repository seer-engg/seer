"""
Meetings API — list org meetings, view transcripts and recordings.

Proxies the external Attendee service with org-level filtering.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from seer.services.meetings import meetings_service

router = APIRouter(prefix="/meetings", tags=["meetings"])


# =============================================================================
# Response models
# =============================================================================


class MeetingListItem(BaseModel):
    bot_id: str
    state: str
    meeting_url: Optional[str] = None
    join_at: Optional[str] = None
    transcription_state: Optional[str] = None
    recording_state: Optional[str] = None
    created_at: Optional[str] = None


class MeetingsListResponse(BaseModel):
    items: list[MeetingListItem]


class MeetingDetailResponse(MeetingListItem):
    events: list[dict] = []
    metadata: dict = {}


class TranscriptUtterance(BaseModel):
    speaker: str
    text: str
    start_ms: int
    duration_ms: int


class TranscriptResponse(BaseModel):
    bot_id: str
    utterances: list[TranscriptUtterance]


class RecordingResponse(BaseModel):
    url: Optional[str] = None
    start_timestamp_ms: Optional[int] = None


# =============================================================================
# Helpers
# =============================================================================


def _require_user(request: Request):
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _get_org_id(request: Request) -> Optional[int]:
    org = getattr(request.state, "organization", None)
    return org.id if org else None


# =============================================================================
# Endpoints
# =============================================================================


@router.get("", response_model=MeetingsListResponse)
async def list_meetings(request: Request) -> MeetingsListResponse:
    """List all meetings for the current organization."""
    _require_user(request)
    org_id = _get_org_id(request)
    if not org_id:
        return MeetingsListResponse(items=[])

    items = await meetings_service.list_meetings(org_id)
    return MeetingsListResponse(items=[MeetingListItem(**m) for m in items])


@router.get("/{bot_id}", response_model=MeetingDetailResponse)
async def get_meeting(request: Request, bot_id: str) -> MeetingDetailResponse:
    """Get detail for a single meeting."""
    _require_user(request)
    org_id = _get_org_id(request)
    if not org_id:
        raise HTTPException(status_code=404, detail="Meeting not found")

    detail = await meetings_service.get_meeting_detail(org_id, bot_id)
    return MeetingDetailResponse(**detail)


@router.get("/{bot_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(request: Request, bot_id: str) -> TranscriptResponse:
    """Get transcript for a meeting."""
    _require_user(request)
    org_id = _get_org_id(request)
    if not org_id:
        raise HTTPException(status_code=404, detail="Meeting not found")

    utterances = await meetings_service.get_meeting_transcript(org_id, bot_id)
    return TranscriptResponse(
        bot_id=bot_id,
        utterances=[TranscriptUtterance(**u) for u in utterances],
    )


@router.get("/{bot_id}/recording", response_model=RecordingResponse)
async def get_recording(request: Request, bot_id: str) -> RecordingResponse:
    """Get recording URL for a meeting (short-lived, fetch on demand)."""
    _require_user(request)
    org_id = _get_org_id(request)
    if not org_id:
        raise HTTPException(status_code=404, detail="Meeting not found")

    recording = await meetings_service.get_meeting_recording_url(org_id, bot_id)
    return RecordingResponse(**recording)


__all__ = ["router"]
