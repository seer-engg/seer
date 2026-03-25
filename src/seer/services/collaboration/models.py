from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CollaborationEventType(str, Enum):
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_UPDATED = "workflow.updated"
    WORKFLOW_DRAFT_UPDATED = "workflow.draft.updated"
    WORKFLOW_PUBLISHED = "workflow.published"
    WORKFLOW_UNPUBLISHED = "workflow.unpublished"
    WORKFLOW_ACTIVE_CHANGED = "workflow.active.changed"
    WORKFLOW_VERSION_RESTORED = "workflow.version.restored"
    WORKFLOW_DELETED = "workflow.deleted"

    ORGANIZATION_UPDATED = "organization.updated"
    MEMBER_ADDED = "organization.member.added"
    MEMBER_REMOVED = "organization.member.removed"
    MEMBER_ROLE_UPDATED = "organization.member.role.updated"
    INVITATION_CREATED = "organization.invitation.created"
    INVITATION_ACCEPTED = "organization.invitation.accepted"
    INVITATION_REVOKED = "organization.invitation.revoked"
    APPROVAL_REQUESTED = "workflow.approval.requested"
    APPROVAL_REVIEWED = "workflow.approval.reviewed"
    INTEGRATION_SHARED = "organization.integration.shared"
    INTEGRATION_UNSHARED = "organization.integration.unshared"

    WORKFLOW_LOCK_ACQUIRED = "workflow.lock.acquired"
    WORKFLOW_LOCK_RELEASED = "workflow.lock.released"
    WORKFLOW_LOCK_EXPIRED = "workflow.lock.expired"
    SYNC_REQUIRED = "sync.required"


class CollaborationEvent(BaseModel):
    event_type: CollaborationEventType
    organization_id: int
    actor_clerk_user_id: str | None = None
    actor_db_user_id: int | None = None
    resource_type: str
    resource_id: str | None = None
    occurred_at: datetime = Field(default_factory=_now_utc)
    correlation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowLockMetadata(BaseModel):
    organization_id: int
    workflow_id: str
    holder_clerk_user_id: str | None = None
    holder_db_user_id: int | None = None
    holder_name: str | None = None
    acquired_at: datetime = Field(default_factory=_now_utc)
    expires_at: datetime
    tab_id: str | None = None
    sse_connection_id: str | None = None  # Format: "{org_id}:{clerk_user_id}:{tab_id}"
