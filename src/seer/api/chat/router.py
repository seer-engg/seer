"""API router for general chat."""

from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from seer.database import User
from seer.database.chat_models import GeneralChatMessage
from seer.database.workflow_models import ChatExecutionStatus

from seer.logger import get_logger

from . import schemas, services

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    return user


@router.post(
    "/sessions",
    response_model=schemas.ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(request: Request, body: schemas.ChatSessionCreate):
    user = _require_user(request)
    session = await services.create_session(user, title=body.title)
    return schemas.ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        current_execution_status=None,
    )


@router.get("/sessions", response_model=List[schemas.ChatSessionResponse])
async def list_sessions(request: Request, limit: int = 50, offset: int = 0):
    user = _require_user(request)
    sessions = await services.list_sessions(user, limit=limit, offset=offset)
    return [
        schemas.ChatSessionResponse(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            current_execution_status=s.current_execution_status.value
            if s.current_execution_status
            else None,
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=schemas.ChatSessionDetailResponse)
async def get_session(request: Request, session_id: int):
    user = _require_user(request)
    session, messages = await services.get_session_with_messages(session_id, user)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return schemas.ChatSessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        current_execution_status=session.current_execution_status.value
        if session.current_execution_status
        else None,
        messages=[
            schemas.ChatMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                model=m.model,
                image_urls=m.image_urls,
                thinking=m.thinking,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(request: Request, session_id: int):
    user = _require_user(request)
    if not await services.delete_session(session_id, user):
        raise HTTPException(status_code=404, detail="Session not found")


@router.patch("/sessions/{session_id}", response_model=schemas.ChatSessionResponse)
async def rename_session(
    request: Request, session_id: int, body: schemas.ChatSessionUpdate
):
    user = _require_user(request)
    session = await services.update_session_title(session_id, user, body.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return schemas.ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        current_execution_status=session.current_execution_status.value
        if session.current_execution_status
        else None,
    )


@router.post("/sessions/{session_id}/messages", response_model=schemas.ChatSendResponse)
async def send_message(
    request: Request, session_id: int, body: schemas.ChatSendRequest
):
    user = _require_user(request)
    session = await services.get_session(session_id, user)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    await services.save_message(
        session_id=session_id, role="user", content=body.message, model=body.model
    )
    await services.auto_generate_title(session_id)

    # Queue background task
    from seer.worker.tasks.general_chat import general_chat_task  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    session.current_execution_status = ChatExecutionStatus.QUEUED
    session.current_execution_error = None
    await session.save(
        update_fields=["current_execution_status", "current_execution_error"]
    )

    task = await general_chat_task.kiq(
        session_id=session_id,
        message=body.message,
        model=body.model,
        user_id=user.id,
        generate_image=body.generate_image,
        image_model=body.image_model,
        image_size=body.image_size or "1024x1024",
        user_timezone=body.timezone,
    )

    session.current_execution_task_id = (
        str(task.task_id) if hasattr(task, "task_id") else None
    )
    await session.save(update_fields=["current_execution_task_id"])

    return schemas.ChatSendResponse(session_id=session_id, execution_status="queued")


@router.post("/sessions/{session_id}/messages/stream")
async def stream_message(
    request: Request, session_id: int, body: schemas.ChatSendRequest
):
    """Stream chat response via SSE. Falls back to task queue for image generation."""
    user = _require_user(request)
    session = await services.get_session(session_id, user)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Image generation doesn't stream — use existing task queue
    if body.generate_image:
        return await send_message(request, session_id, body)

    # Save user message + auto-title
    await services.save_message(
        session_id=session_id, role="user", content=body.message, model=body.model
    )
    await services.auto_generate_title(session_id)

    async def event_generator():
        try:
            from seer.config import config  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
            from seer.llm import get_llm  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
            from seer.worker.tasks.general_chat import _build_langchain_messages  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

            from langchain_core.messages import SystemMessage  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
            from seer.prompts import get_datetime_context  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

            history = (
                await GeneralChatMessage.filter(session_id=session_id)
                .order_by("created_at")
                .all()
            )
            lc_messages = [
                SystemMessage(content=get_datetime_context(user_timezone=body.timezone))
            ] + _build_langchain_messages(history)

            model = body.model or config.default_llm_model
            llm = get_llm(model=model)

            full_response = ""
            async for chunk in llm.astream(lc_messages):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            # Save complete assistant message
            await services.save_message(
                session_id=session_id,
                role="assistant",
                content=full_response,
                model=model,
            )
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: SSE must send error event, not crash
            logger.error(
                "Streaming chat failed", exc_info=True, extra={"session_id": session_id}
            )
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/sessions/{session_id}/status", response_model=schemas.ChatStatusResponse)
async def get_status(request: Request, session_id: int):
    user = _require_user(request)
    session = await services.get_session(session_id, user)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    response_text = None
    image_urls = None
    thinking = None

    if session.current_execution_status in (
        ChatExecutionStatus.COMPLETED,
        ChatExecutionStatus.FAILED,
    ):
        last_msg = (
            await GeneralChatMessage.filter(session=session, role="assistant")
            .order_by("-created_at")
            .first()
        )
        if last_msg:
            response_text = last_msg.content
            image_urls = last_msg.image_urls
            thinking = last_msg.thinking

    return schemas.ChatStatusResponse(
        status=session.current_execution_status.value
        if session.current_execution_status
        else None,
        response=response_text,
        image_urls=image_urls,
        thinking=thinking,
        error=session.current_execution_error,
    )
