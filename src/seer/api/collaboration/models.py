from __future__ import annotations

from pydantic import BaseModel, Field

from seer.services.collaboration.models import WorkflowLockMetadata


class WorkflowLockAcquireRequest(BaseModel):
    tab_id: str | None = Field(None, max_length=255)


class WorkflowLockResponse(BaseModel):
    lock: WorkflowLockMetadata


class WorkflowLockStatusResponse(BaseModel):
    lock: WorkflowLockMetadata | None = None
