from __future__ import annotations

from typing import AsyncIterator, Optional

from fastapi import APIRouter, Body, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from seer.api.collaboration.models import (
    WorkflowLockAcquireRequest,
    WorkflowLockHeartbeatRequest,
    WorkflowLockResponse,
    WorkflowLockStatusResponse,
)
from seer.api.core.middleware.organization import get_membership, get_organization
from seer.api.workflows.services.shared import _get_workflow_org_scoped
from seer.database import User
from seer.services.collaboration import (
    CollaborationEventType,
    WorkflowLockService,
    publish_collaboration_event,
)

from seer.api.collaboration.sse import stream_org_events_sse

router = APIRouter(prefix="/collaboration", tags=["collaboration"])


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


@router.get("/events")
async def stream_collaboration_events(
    request: Request,
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    user = _require_user(request)
    del user  # auth check only
    organization = get_organization(request)
    membership = get_membership(request)
    if membership.status.value != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive organization membership")

    async def _should_stop() -> bool:
        if await request.is_disconnected():
            return True
        shutdown_event = getattr(request.app.state, "shutdown_event", None)
        return bool(shutdown_event and shutdown_event.is_set())

    async def _event_stream() -> AsyncIterator[str]:
        async for chunk in stream_org_events_sse(
            organization.id,
            last_event_id=last_event_id,
            should_stop=_should_stop,
        ):
            yield chunk

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/workflows/{workflow_id}/lock", response_model=WorkflowLockStatusResponse)
async def get_workflow_lock(request: Request, workflow_id: str) -> WorkflowLockStatusResponse:
    user = _require_user(request)
    organization = get_organization(request)
    membership = get_membership(request)
    await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=False)

    service = WorkflowLockService()
    try:
        return WorkflowLockStatusResponse(lock=await service.get_lock(organization.id, workflow_id))
    finally:
        await service.close()


@router.post("/workflows/{workflow_id}/lock", response_model=WorkflowLockResponse)
async def acquire_workflow_lock(
    request: Request,
    workflow_id: str,
    payload: WorkflowLockAcquireRequest = Body(default_factory=WorkflowLockAcquireRequest),
) -> WorkflowLockResponse:
    user = _require_user(request)
    organization = get_organization(request)
    membership = get_membership(request)
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)

    service = WorkflowLockService()
    try:
        acquired, lock = await service.acquire_lock(
            organization_id=organization.id,
            workflow_id=workflow.workflow_id,
            user=user,
            tab_id=payload.tab_id,
        )
    finally:
        await service.close()

    if not acquired:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=WorkflowLockResponse(lock=lock).model_dump(mode="json"),
        )

    await publish_collaboration_event(
        organization_id=organization.id,
        actor=user,
        event_type=CollaborationEventType.WORKFLOW_LOCK_ACQUIRED,
        resource_type="workflow",
        resource_id=workflow.workflow_id,
        correlation_id=getattr(request.state, "correlation_id", None),
        payload={"tab_id": lock.tab_id, "holder_name": lock.holder_name},
    )
    return WorkflowLockResponse(lock=lock)


@router.post("/workflows/{workflow_id}/lock/heartbeat", response_model=WorkflowLockResponse)
async def heartbeat_workflow_lock(
    request: Request,
    workflow_id: str,
    payload: WorkflowLockHeartbeatRequest = Body(default_factory=WorkflowLockHeartbeatRequest),
) -> WorkflowLockResponse:
    user = _require_user(request)
    organization = get_organization(request)
    membership = get_membership(request)
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)

    service = WorkflowLockService()
    try:
        lock = await service.heartbeat_lock(
            organization_id=organization.id,
            workflow_id=workflow.workflow_id,
            user=user,
            tab_id=payload.tab_id,
        )
    finally:
        await service.close()

    if lock is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Workflow lock not held by current user")
    return WorkflowLockResponse(lock=lock)


@router.delete("/workflows/{workflow_id}/lock", status_code=status.HTTP_204_NO_CONTENT)
async def release_workflow_lock(
    request: Request,
    workflow_id: str,
    tab_id: str | None = Query(None),
) -> None:
    user = _require_user(request)
    organization = get_organization(request)
    membership = get_membership(request)
    workflow = await _get_workflow_org_scoped(user, workflow_id, organization, membership, require_manage=True)

    service = WorkflowLockService()
    try:
        released_lock = await service.release_lock(
            organization_id=organization.id,
            workflow_id=workflow.workflow_id,
            user=user,
            tab_id=tab_id,
        )
    finally:
        await service.close()

    if released_lock is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Workflow lock not held by current user")

    await publish_collaboration_event(
        organization_id=organization.id,
        actor=user,
        event_type=CollaborationEventType.WORKFLOW_LOCK_RELEASED,
        resource_type="workflow",
        resource_id=workflow.workflow_id,
        correlation_id=getattr(request.state, "correlation_id", None),
        payload={"tab_id": released_lock.tab_id, "holder_name": released_lock.holder_name},
    )
