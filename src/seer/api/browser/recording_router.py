# pylint: disable=broad-exception-caught
# Reason: API router needs graceful error handling for user-facing responses
"""
Session recording API endpoints.

Provides endpoints for listing, viewing, and deleting browser session recordings.
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from seer.database import User
from seer.database.models_browser_recording import SessionRecording
from seer.logger import get_logger
from seer.services.browser.recording_service import RecordingService

logger = get_logger("api.browser.recording_router")

router = APIRouter(prefix="/browser/recordings", tags=["browser-recordings"])


# ------------------------------------------------------------------
# Response Models
# ------------------------------------------------------------------
class RecordingMetadata(BaseModel):
    """Recording metadata without event data."""
    id: str
    session_type: str
    event_count: int
    duration_ms: int
    compressed_size_bytes: int
    start_url: Optional[str]
    status: str
    created_at: Optional[str]
    completed_at: Optional[str]
    browser_profile_id: Optional[str]
    workflow_run_id: Optional[str]
    is_chunked: bool = False
    chunk_count: int = 0


class RecordingListResponse(BaseModel):
    """Paginated list of recordings."""
    recordings: List[RecordingMetadata]
    total: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@router.get("", response_model=RecordingListResponse)
async def list_recordings(
    request: Request,
    profile_id: Optional[UUID] = Query(default=None, description="Filter by browser profile"),
    workflow_run_id: Optional[str] = Query(default=None, description="Filter by workflow run ID"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> RecordingListResponse:
    """List session recordings for the current user."""
    user: User = request.state.db_user

    filters = {"user": user}
    if profile_id:
        filters["browser_profile_id"] = str(profile_id)
    if workflow_run_id:
        filters["workflow_run_id"] = workflow_run_id

    total = await SessionRecording.filter(**filters).count()
    recordings = (
        await SessionRecording.filter(**filters)
        .offset(offset)
        .limit(limit)
        .all()
    )

    return RecordingListResponse(
        recordings=[
            RecordingMetadata(
                id=str(r.id),
                session_type=r.session_type,
                event_count=r.event_count,
                duration_ms=r.duration_ms,
                compressed_size_bytes=r.compressed_size_bytes,
                start_url=r.start_url,
                status=r.status,
                created_at=r.created_at.isoformat() if r.created_at else None,
                completed_at=r.completed_at.isoformat() if r.completed_at else None,
                browser_profile_id=str(r.browser_profile_id) if r.browser_profile_id else None,
                workflow_run_id=r.workflow_run_id,
                is_chunked=r.is_chunked,
                chunk_count=r.chunk_count,
            )
            for r in recordings
        ],
        total=total,
    )


@router.get("/{recording_id}", response_model=RecordingMetadata)
async def get_recording(
    request: Request,
    recording_id: UUID,
) -> RecordingMetadata:
    """Get recording metadata (without event data)."""
    user: User = request.state.db_user

    recording = await SessionRecording.get_or_none(id=recording_id, user=user)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    return RecordingMetadata(
        id=str(recording.id),
        session_type=recording.session_type,
        event_count=recording.event_count,
        duration_ms=recording.duration_ms,
        compressed_size_bytes=recording.compressed_size_bytes,
        start_url=recording.start_url,
        status=recording.status,
        created_at=recording.created_at.isoformat() if recording.created_at else None,
        completed_at=recording.completed_at.isoformat() if recording.completed_at else None,
        browser_profile_id=str(recording.browser_profile_id) if recording.browser_profile_id else None,
        workflow_run_id=recording.workflow_run_id,
        is_chunked=recording.is_chunked,
        chunk_count=recording.chunk_count,
    )


@router.get("/{recording_id}/events")
async def get_recording_events(
    request: Request,
    recording_id: UUID,
) -> dict:
    """Get decompressed rrweb events for a recording.

    Handles both chunked and non-chunked recordings transparently.
    For chunked recordings, reassembles events from all chunks in order.
    """
    user: User = request.state.db_user

    recording = await SessionRecording.get_or_none(id=recording_id, user=user)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    try:
        # Use service method that handles both chunked and non-chunked
        events = await RecordingService.get_recording_events(str(recording_id))
    except Exception as e:
        logger.error("Failed to get recording events %s: %s", recording_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve recording events") from e

    return {
        "recording_id": str(recording_id),
        "event_count": len(events),
        "events": events,
        "is_chunked": recording.is_chunked,
        "chunk_count": recording.chunk_count,
    }


@router.get("/shared/{recording_id}/events")
async def get_shared_recording_events(recording_id: UUID) -> dict:
    """Get recording events for public sharing (no auth required).

    Handles both chunked and non-chunked recordings transparently.
    """
    recording = await SessionRecording.get_or_none(id=recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    try:
        # Use service method that handles both chunked and non-chunked
        events = await RecordingService.get_recording_events(str(recording_id))
    except Exception as e:
        logger.error("Failed to get recording events %s: %s", recording_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve recording events") from e

    return {
        "recording_id": str(recording_id),
        "event_count": len(events),
        "events": events,
        "is_chunked": recording.is_chunked,
        "chunk_count": recording.chunk_count,
    }


@router.delete("/{recording_id}")
async def delete_recording(
    request: Request,
    recording_id: UUID,
) -> dict:
    """Delete a session recording."""
    user: User = request.state.db_user

    recording = await SessionRecording.get_or_none(id=recording_id, user=user)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    await recording.delete()
    logger.info("Deleted recording %s for user %s", recording_id, user.user_id)

    return {"deleted": True, "recording_id": str(recording_id)}
