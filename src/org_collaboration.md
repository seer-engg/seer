# Organization Collaboration Implementation

This document describes a concrete backend implementation for real-time organization collaboration in Seer.

The target is the current codebase, not a greenfield design:

- API routes are mounted under `src/seer/api/` and exposed through `src/seer/api/router.py`
- active org context comes from `src/seer/api/core/middleware/organization.py`
- workflow mutations live in `src/seer/api/workflows/services/lifecycle.py`
- many organization mutations still live directly in `src/seer/api/organizations/router.py`
- Redis is already part of the stack
- Seer already uses `Redis Streams + SSE` for agent streaming in `src/seer/agents/nexus/stream_publisher.py` and `src/seer/api/agents/workflow/sse.py`

## Goals

- notify other organization members when org-scoped data changes
- keep multi-user screens fresh without polling every page aggressively
- support a simple single-editor workflow lock for the canvas
- work correctly across multiple API instances

## Non-goals

- CRDT-style collaborative document editing
- broadcasting entire workflow specs over the event bus
- exact-once delivery guarantees

## 1. Design Constraints From This Repo

Seer has a strict dependency direction:

```text
api -> services -> core -> tools
         |
       worker
```

That means collaboration primitives used by `services/` or `worker/` cannot live under `api/`.

The original draft placed shared collaboration models under `src/seer/api/collaboration/`. That would force upward imports from `services/` into `api/`, which violates the repo rules. The implementation should instead look like this:

- `src/seer/services/collaboration/models.py`
- `src/seer/services/collaboration/publisher.py`
- `src/seer/services/collaboration/lock_service.py`
- `src/seer/api/collaboration/router.py`
- `src/seer/api/collaboration/sse.py`

If common low-level code later needs to be reused outside `services/`, extract only the narrow abstraction downward. Do not start by putting Redis collaboration code in `core/`.

## 2. Core Approach

Use two mechanisms:

1. `Redis Streams + SSE` for org-scoped invalidation events
2. `Redis lease locks with TTL` for workflow edit ownership

Events should be small and typed. They are invalidation signals, not document sync payloads.

Clients decide whether to:

- invalidate and refetch
- show a banner
- switch the workflow canvas to read-only
- refresh only the visible workflow detail

## 3. Why Redis Streams

Use `Redis Streams`, not plain Pub/Sub, for org change notifications.

Reasons:

- SSE reconnect already supports `Last-Event-ID`
- Redis Stream IDs map naturally to `Last-Event-ID`
- reconnects after browser refresh, ALB connection drops, or ECS reschedules are manageable
- any API instance can publish while any other API instance serves the SSE connection

This matches the pattern Seer already uses for agent streaming.

## 4. Scope of Events

Start with org-scoped changes that already affect multiple users:

- workflow created
- workflow updated
- workflow draft patched
- workflow published or unpublished
- workflow active flag changed
- workflow version restored
- workflow deleted
- organization metadata updated
- member added, removed, or role changed
- invitation created, accepted, or revoked
- approval requested or reviewed
- integration shared or unshared

Lock-related events can share the same org stream:

- `workflow.lock.acquired`
- `workflow.lock.released`
- `workflow.lock.expired`

Skip `workflow.lock.heartbeat` at first unless the frontend proves it needs live presence updates. Heartbeats are primarily for TTL renewal, not UI fan-out.

## 5. Event Model

Keep the shared event contract in `src/seer/services/collaboration/models.py`.

Suggested shape:

```python
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
```

Guidelines:

- keep `payload` small
- send identifiers and small routing hints only
- never publish the full workflow spec
- prefer string resource IDs exactly as the frontend already sees them, for example `wf_123`

## 6. Redis Keys

Use per-organization stream keys:

- `org:events:{organization_id}`

Use per-workflow lock keys:

- `org:{organization_id}:workflow:{workflow_id}:lock`

Suggested TTLs:

- stream TTL: `7200` seconds to start
- lock TTL: `30` seconds
- heartbeat interval: `10` seconds

The stream TTL can be extended later if reconnect windows prove too small.

## 7. Publisher Implementation

Build an org-scoped publisher that mirrors `src/seer/agents/nexus/stream_publisher.py`, but place it in `services/`, not `api/`.

Suggested interface:

```python
class OrgEventPublisher:
    def __init__(self, organization_id: int):
        self.organization_id = organization_id
        self.stream_key = f"org:events:{organization_id}"

    async def publish(self, event: CollaborationEvent) -> str | None:
        ...
```

Behavior:

- lazy-create the Redis client exactly like existing stream publishers
- `XADD` JSON payloads into the org stream
- refresh TTL after each publish
- return the Redis message ID on success
- never fail the primary mutation because publishing failed

Use `seer.logger.get_logger`, not `print()` and not ad hoc `logging.getLogger()`.

## 8. SSE Endpoint

Add a thin API router:

- `src/seer/api/collaboration/router.py`

Expose:

- `GET /api/collaboration/events`

Do not accept `organization_id` as a query parameter. Use the active org already attached by middleware:

- `request.state.db_user`
- `request.state.organization`
- `request.state.membership`

That preserves org isolation and matches the rest of the API.

Implementation notes:

- require an authenticated user
- require an active organization and membership
- read the `Last-Event-ID` header
- stream events from `org:events:{organization_id}`
- emit SSE heartbeat comments every 25 to 30 seconds

Suggested wire format:

```text
id: 1710000000000-0
event: collaboration
data: {"event_type":"workflow.updated","organization_id":12,...}

```

Using `event: collaboration` is optional, but it makes browser-side handling slightly cleaner.

## 9. SSE Read Behavior

Reuse the design of `src/seer/api/agents/workflow/sse.py` with two deliberate differences:

- for org collaboration, first connect should usually start at `$` so the client gets new events only
- reconnects should resume from `Last-Event-ID`

Recommended behavior:

- first connect, no `Last-Event-ID`: start at `$`
- reconnect with `Last-Event-ID`: start after that message ID
- if the stream no longer contains the requested event window, emit a lightweight `sync.required` event or rely on the client to perform a full refetch on reconnect

The simplest first cut is:

- resume when possible
- on reconnect, the client also invalidates key org-level queries once

That gives correctness without overbuilding stream-history recovery.

## 10. Lock Service

Put lock logic in `src/seer/services/collaboration/lock_service.py`.

Endpoints:

- `POST /api/collaboration/workflows/{workflow_id}/lock`
- `POST /api/collaboration/workflows/{workflow_id}/lock/heartbeat`
- `DELETE /api/collaboration/workflows/{workflow_id}/lock`
- `GET /api/collaboration/workflows/{workflow_id}/lock`

Suggested stored value:

```json
{
  "organization_id": 12,
  "workflow_id": "wf_123",
  "holder_clerk_user_id": "user_abc",
  "holder_db_user_id": 45,
  "holder_name": "Jane Doe",
  "acquired_at": "2026-03-18T12:00:00Z",
  "expires_at": "2026-03-18T12:00:30Z",
  "tab_id": "a3f2"
}
```

Acquire logic:

- use `SET key value NX EX 30`
- if successful, return the lock payload and publish `workflow.lock.acquired`
- if not successful, load the current lock value and return `409 Conflict`

Heartbeat logic:

- only the current holder may heartbeat
- validate `holder_clerk_user_id` or `holder_db_user_id`
- validate `tab_id` if the client sends one
- reset TTL to 30 seconds

Release logic:

- only the current holder may release
- delete the key
- publish `workflow.lock.released`

Natural expiry:

- no sweeper is required in phase 1
- the next acquire attempt can treat a missing key as expired
- `workflow.lock.expired` is optional and can be emitted only when useful

## 11. Lock Semantics

The lock is for the workflow editor, not for general page viewing.

Recommended policy:

- viewing a workflow does not acquire a lock
- entering edit mode, or the first draft mutation, acquires the lock
- other org members see the workflow as read-only while the lock is held
- publish, rename, and draft mutation paths should require lock ownership for team workflows
- personal workspaces can skip lock enforcement entirely if desired

This is intentionally a lease, not a permanent mutex.

## 12. Where To Publish Events

Publish at shared mutation boundaries after successful DB writes.

### Workflow events

Primary hooks belong in:

- `src/seer/api/workflows/services/lifecycle.py`

Relevant functions already present:

- `create_workflow`
- `update_workflow`
- `patch_workflow_draft`
- `toggle_workflow_published`
- `toggle_workflow_active`
- `restore_workflow_version`
- `publish_workflow`
- `delete_workflow`
- `import_workflow` if imported workflows should appear live to teammates

### Organization and membership events

Today, many of these mutations still live directly in:

- `src/seer/api/organizations/router.py`

That is not ideal architecturally, but it is the real current hook point for a first implementation. Publish after successful completion for:

- org update
- member role update
- member removal
- invitation create
- invitation accept
- invitation revoke
- workflow transfer to org
- approval request
- approval review
- integration share
- integration unshare

If this area grows, the right follow-up is to extract those mutation paths into `services/organization_*` helpers and keep the router thin.

## 13. Transaction Timing

Rules:

- publish only after the DB mutation succeeds
- if a mutation is wrapped in a transaction, publish after the commit boundary
- event publication is best-effort and must not roll back the primary mutation

If stronger guarantees are required later, add an outbox table. Do not start there.

## 14. Permissions

All collaboration access is org-scoped.

SSE endpoint requirements:

- authenticated user
- active membership in the current org

Lock endpoint requirements:

- authenticated user
- active membership in the current org
- workflow belongs to the active org
- caller has the same workflow visibility or manage permission checks already used by workflow services

Use the existing workflow-org scoping helpers from `src/seer/api/workflows/services/shared.py` instead of inventing a second permission model.

## 15. Client Routing Guidance

Keep server events resource-oriented. The backend should not know about React Query cache keys.

Suggested client handling:

- `workflow.*` invalidates workflow list, detail, and version queries
- `organization.member.*` refreshes member lists
- `organization.invitation.*` refreshes invitations
- `organization.updated` refreshes org header and settings
- `workflow.approval.*` refreshes approvals
- `organization.integration.*` refreshes org-shared integration lists
- `workflow.lock.*` updates editor lock state and read-only banners

## 16. ALB and Heartbeats

SSE connections will be dropped by intermediaries if they sit idle too long.

Server requirements:

- emit `: heartbeat` comments every 25 to 30 seconds
- set `Content-Type: text/event-stream`
- set `Cache-Control: no-cache`
- set `Connection: keep-alive`
- set `X-Accel-Buffering: no` if any buffering proxy is involved

Infra requirement:

- ALB idle timeout must be higher than the heartbeat interval

## 17. Observability

Add structured logs for:

- SSE connection opened and closed
- org ID on each collaboration stream
- publish failures by event type
- lock acquire success and conflict
- unauthorized heartbeat or release attempts

Useful counters:

- collaboration events published by event type
- collaboration publish failures
- active collaboration SSE connections
- lock acquisition conflicts

Use `seer.logger.get_logger` for logs. If metrics are added, keep them lightweight and tagged by event type and org where safe.

## 18. Testing Plan

### Unit tests

Add tests for:

- event model serialization
- publisher `XADD` behavior
- lock acquisition success and conflict
- lock heartbeat authorization
- lock release authorization

### Integration tests

Add tests for:

- workflow update publishes an org event
- invitation creation publishes an org event
- invitation acceptance publishes an org event
- SSE reconnect resumes from `Last-Event-ID`
- lock acquire on one app instance is visible to another through Redis

### Important repo-specific rule

This change is a bug-prone shared behavior change. If any workflow mutation path is changed, add tests around the exact lifecycle function touched. Do not rely only on manual testing.

## 19. Recommended Rollout

### Phase 1

- add collaboration event models
- add Redis Stream publisher
- add `/api/collaboration/events`
- publish only workflow, member, invitation, and org update invalidation events

### Phase 2

- add workflow lock endpoints and UI lock banner
- enforce the lock for draft-editing paths on team workflows

### Phase 3

- improve reconnect recovery
- add `sync.required` if needed
- add dashboards and alerting if collaboration traffic becomes operationally important

## 20. Smallest Valuable First Cut

If the goal is to ship freshness quickly without overbuilding, implement exactly this subset first:

- one org-scoped SSE endpoint
- one Redis Stream per org
- publish only:
  - `workflow.created`
  - `workflow.updated`
  - `workflow.draft.updated`
  - `workflow.deleted`
  - `organization.member.added`
  - `organization.member.removed`
  - `organization.member.role.updated`
  - `organization.invitation.created`
  - `organization.invitation.accepted`
- client invalidates and refetches
- reconnect triggers one defensive full refetch

Then add workflow edit locks once the invalidation path is stable.
