# pylint: disable=broad-exception-caught,logging-fstring-interpolation
# Reason: WebSocket/session endpoints require flexible exception handling and dynamic logging
"""
WebSocket streaming and session lifecycle endpoints for browser automation.

Provides REST endpoints for creating/completing interactive browser sessions,
and a WebSocket endpoint for live screen streaming + input dispatch.
"""
from __future__ import annotations

import asyncio
import base64
import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from seer.auth.clerk_verifier import ClerkJWTVerifier
from seer.config import config
from seer.database import User
from seer.logger import get_logger
from seer.services.browser.pool_manager import BrowserPoolManager, ManagedSession
from seer.services.browser.profile_manager import BrowserProfileManager
from seer.services.browser.streaming_service import StreamingService

logger = get_logger("api.browser.ws_router")


# ------------------------------------------------------------------
# WebSocket Input Handlers
# ------------------------------------------------------------------
async def _handle_mouse_event(streamer: StreamingService, data: dict) -> None:
    """Handle mouse input events from WebSocket client."""
    action = data.get("action", "click")
    x = float(data.get("x", 0))
    y = float(data.get("y", 0))
    button = data.get("button", "left")
    click_count = int(data.get("clickCount", 1))

    if action == "click":
        # Use CDP mousePressed/mouseReleased sequence for trusted clicks
        # that propagate into cross-origin iframes (e.g. reCAPTCHA)
        await streamer.dispatch_click_js(x, y, button)
    elif action == "down":
        await streamer.dispatch_mouse_event("mousePressed", x, y, button=button, click_count=click_count)
    elif action == "up":
        await streamer.dispatch_mouse_event("mouseReleased", x, y, button=button, click_count=click_count)
    elif action == "move":
        await streamer.dispatch_mouse_event("mouseMoved", x, y)


async def _handle_key_event(streamer: StreamingService, data: dict) -> None:
    """Handle keyboard input events from WebSocket client."""
    action = data.get("action", "press")
    key = data.get("key", "")
    code = data.get("code", "")
    text = data.get("text", "")

    if action == "press":
        await streamer.dispatch_key_event("keyDown", key, code=code, text=text)
        if text:
            await streamer.dispatch_key_event("char", key, code=code, text=text)
        await streamer.dispatch_key_event("keyUp", key, code=code)
    elif action == "down":
        await streamer.dispatch_key_event("keyDown", key, code=code, text=text)
    elif action == "up":
        await streamer.dispatch_key_event("keyUp", key, code=code)


async def _handle_scroll_event(streamer: StreamingService, data: dict) -> None:
    """Handle scroll input events from WebSocket client."""
    x = float(data.get("x", 0))
    y = float(data.get("y", 0))
    delta_x = float(data.get("deltaX", 0))
    delta_y = float(data.get("deltaY", -120))
    await streamer.dispatch_scroll_event(x, y, delta_x, delta_y)


async def _handle_navigate_event(managed: ManagedSession, data: dict) -> None:
    """Handle navigation requests from WebSocket client."""
    url = data.get("url", "")
    if not url:
        return
    try:
        page = await managed.session.must_get_current_page()
        # pylint: disable=protected-access
        # Reason: browser-use stores target_id as private _target_id (no public property)
        cdp_session = await managed.session.get_or_create_cdp_session(page._target_id)
        await managed.session.cdp_client.send.Page.navigate(
            params={"url": url},
            session_id=cdp_session.session_id,
        )
    except Exception as e:
        logger.warning(f"Navigate failed: {e}")


async def _validate_ws_session(
    websocket: WebSocket, session_id: str, token: str
) -> Optional[ManagedSession]:
    """Validate WebSocket auth and session ownership.

    Returns ManagedSession if valid, None otherwise (websocket already closed).
    """
    user_id = await _verify_ws_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Authentication failed")
        return None

    pool = await BrowserPoolManager.get_instance()
    managed = pool.get_session(session_id)
    if not managed:
        await websocket.close(code=4004, reason="Session not found")
        return None
    if managed.user_id != user_id:
        await websocket.close(code=4003, reason="Session does not belong to this user")
        return None

    return managed


async def _dispatch_input_event(
    streamer: StreamingService, managed: Optional[ManagedSession], data: dict
) -> None:
    """Dispatch a single input event to the appropriate handler."""
    msg_type = data.get("type")
    if msg_type == "mouse":
        await _handle_mouse_event(streamer, data)
    elif msg_type == "key":
        await _handle_key_event(streamer, data)
    elif msg_type == "scroll":
        await _handle_scroll_event(streamer, data)
    elif msg_type == "navigate" and managed is not None:
        await _handle_navigate_event(managed, data)


async def _run_streaming_loop(
    websocket: WebSocket,
    streamer: StreamingService,
    session_id: str,
    managed: Optional[ManagedSession] = None,
) -> None:
    """Run the bidirectional streaming loop for WebSocket communication.

    When managed is None (HITL mode), navigate events are ignored.
    """

    async def send_frames() -> None:
        """Send screencast frames to client."""
        while True:
            frame = await streamer.get_frame(timeout=5.0)
            if frame is not None:
                b64_data = base64.b64encode(frame).decode("ascii")
                await websocket.send_json({
                    "type": "frame",
                    "data": b64_data,
                    "timestamp": time.time(),
                })

    async def receive_input() -> None:
        """Receive and dispatch input events from client."""
        while True:
            data = await websocket.receive_json()
            await _dispatch_input_event(streamer, managed, data)

    try:
        send_task = asyncio.create_task(send_frames())
        receive_task = asyncio.create_task(receive_input())
        _done, pending = await asyncio.wait(
            [send_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")

router = APIRouter(prefix="/browser/sessions", tags=["browser-sessions"])


# ------------------------------------------------------------------
# Request/Response Models
# ------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    """Request to create an interactive browser session."""
    profile_id: UUID
    target_url: Optional[str] = Field(
        default=None,
        description="Optional starting URL for the session"
    )


class SessionResponse(BaseModel):
    """Response from creating an interactive session."""
    session_id: str
    profile_id: str
    ws_url: str
    recording_id: Optional[str] = None
    status: str


class CompleteSessionResponse(BaseModel):
    """Response from completing an interactive session."""
    profile_id: str
    logged_in_domains: list[str]
    recording_id: Optional[str] = None
    status: str


# ------------------------------------------------------------------
# REST Endpoints
# ------------------------------------------------------------------
@router.post("", response_model=SessionResponse)
async def create_interactive_session(
    request: Request,
    body: CreateSessionRequest,
) -> SessionResponse:
    """Create an interactive browser session for streaming login.

    Returns a session ID and WebSocket URL for live streaming.
    """
    user: User = request.state.db_user
    logger.info(
        "Creating interactive session for profile %s, user %s",
        body.profile_id, user.user_id,
    )

    manager = BrowserProfileManager()
    try:
        result = await manager.create_interactive_session(
            user, body.profile_id, body.target_url
        )
    except Exception as e:
        logger.error("Failed to create interactive session: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Build WebSocket URL relative to the request
    ws_url = f"/api/browser/sessions/{result['session_id']}/stream"

    return SessionResponse(
        session_id=result["session_id"],
        profile_id=result["profile_id"],
        ws_url=ws_url,
        recording_id=result.get("recording_id"),
        status=result["status"],
    )


@router.post("/{session_id}/complete", response_model=CompleteSessionResponse)
async def complete_session(
    request: Request,
    session_id: str,
) -> CompleteSessionResponse:
    """Complete an interactive session, saving encrypted state.

    Exports browser state, saves it encrypted to the profile,
    and releases the pool slot.
    """
    user: User = request.state.db_user
    logger.info("Completing session %s for user %s", session_id, user.user_id)

    # Look up the session to get profile_id
    pool = await BrowserPoolManager.get_instance()
    managed = pool.get_session(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")
    if managed.user_id != str(user.user_id):
        raise HTTPException(status_code=403, detail="Session does not belong to this user")

    profile_id = managed.profile_id
    if not profile_id:
        raise HTTPException(status_code=400, detail="Session has no associated profile")

    manager = BrowserProfileManager()
    try:
        result = await manager.complete_interactive_session(
            user, UUID(profile_id), session_id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to complete session: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return CompleteSessionResponse(
        profile_id=result["profile_id"],
        logged_in_domains=result["logged_in_domains"],
        recording_id=result.get("recording_id"),
        status=result["status"],
    )


# ------------------------------------------------------------------
# WebSocket Auth Helper
# ------------------------------------------------------------------
async def _verify_ws_token(token: str) -> Optional[str]:
    """Verify JWT token for WebSocket connection using ClerkJWTVerifier.

    Returns:
        User ID string, or None if verification fails
    """
    if not token:
        return None

    verifier = ClerkJWTVerifier(
        jwks_url=config.clerk_jwks_url,
        issuer=config.clerk_issuer,
        audience=config.clerk_audience,
    )
    verified = verifier.verify_token(token)
    return verified.user_id if verified else None


# ------------------------------------------------------------------
# WebSocket Endpoint
# ------------------------------------------------------------------
async def _start_and_stream(
    websocket: WebSocket,
    streamer: StreamingService,
    session_id: str,
    *,
    managed: Optional[ManagedSession] = None,
    target_id: Optional[str] = None,
) -> None:
    """Start streaming and run the bidirectional loop, then clean up.

    Exactly one of managed or target_id must be provided.
    - managed: normal pool session (start via CDP session)
    - target_id: HITL session (start via raw target ID)
    """
    label = "HITL " if target_id else ""
    try:
        if managed:
            await streamer.start(managed.session)
        else:
            await streamer.start_from_target(target_id)
        await websocket.send_json({"type": "status", "status": "connected", "message": f"{label}Streaming started"})
    except Exception as e:
        logger.error(f"Failed to start {label}streaming for session {session_id}: {e}")
        await websocket.send_json({"type": "error", "code": "cdp_error", "message": str(e)})
        await websocket.close()
        return

    try:
        await _run_streaming_loop(websocket, streamer, session_id, managed=managed)
    finally:
        await streamer.stop()
        try:
            await websocket.send_json({"type": "status", "status": "closed", "message": "Streaming stopped"})
        except Exception:
            pass


def _new_streamer() -> StreamingService:
    """Create a StreamingService with standard screencast settings."""
    return StreamingService(
        quality=config.browser_screencast_quality,
        max_width=config.browser_screencast_max_width,
        max_height=config.browser_screencast_max_height,
        every_nth_frame=config.browser_screencast_every_nth_frame,
    )


@router.websocket("/{session_id}/stream")
async def stream_session(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(default=""),
) -> None:
    """Live browser streaming via WebSocket.

    Auth: JWT passed as ?token= query parameter (WebSocket API cannot set headers).

    Protocol:
        Server → Client: {"type":"frame","data":"<b64-jpeg>","timestamp":...}
        Server → Client: {"type":"status","status":"connected|error|closed","message":"..."}
        Client → Server: {"type":"mouse","action":"click|move","x":...,"y":...}
        Client → Server: {"type":"key","action":"down|up|press","key":"Enter"}
        Client → Server: {"type":"scroll","x":...,"y":...,"deltaY":-120}
        Client → Server: {"type":"navigate","url":"https://..."}
    """
    user_id = await _verify_ws_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Check if this is a normal pool session or a browser_hitl session
    pool = await BrowserPoolManager.get_instance()
    managed = pool.get_session(session_id)

    if managed:
        if managed.user_id != user_id:
            await websocket.close(code=4003, reason="Session does not belong to this user")
            return
        await websocket.accept()
        await _start_and_stream(websocket, _new_streamer(), session_id, managed=managed)
        return

    # HITL path: session is in the worker process
    target_id = await _get_hitl_target_id(session_id, user_id)
    if not target_id:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    await _start_and_stream(websocket, _new_streamer(), session_id, target_id=target_id)


async def _get_hitl_target_id(session_id: str, user_id: str) -> Optional[str]:
    """Look up the CDP target_id for a browser_hitl session from the DB.

    Finds an interrupted workflow run whose pending_interrupt_data contains
    the given session_id, then extracts the target_id.
    """
    from seer.database.workflow_models import WorkflowRun, WorkflowRunStatus  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    # Find the user by clerk_user_id to get user.id for ownership check
    user = await User.get_or_none(clerk_user_id=user_id)
    if not user:
        return None

    # Find interrupted runs for this user with browser_hitl data
    runs = await WorkflowRun.filter(
        user_id=user.id,
        status=WorkflowRunStatus.INTERRUPTED,
    ).all()

    for run in runs:
        interrupt_data = run.pending_interrupt_data
        if not isinstance(interrupt_data, dict):
            continue
        if interrupt_data.get("type") != "browser_hitl":
            continue
        if interrupt_data.get("session_id") == session_id:
            target_id = interrupt_data.get("target_id")
            if target_id:
                logger.info(f"Found HITL target_id={target_id} for session={session_id}")
                return target_id

    logger.warning(f"No HITL target_id found for session={session_id}")
    return None
