# Organization Collaboration Implementation

This document describes a concrete frontend implementation for real-time organization collaboration.

It is written for the current Seer frontend:

- React + TypeScript
- React Query
- Zustand stores
- organization state in `src/stores/organizationStore.ts` and `src/stores/membersStore.ts`
- workflow queries in `src/hooks/useWorkflowQueries.ts`
- workflow local state in `src/stores/workflowStore.ts`
- existing SSE parser in `src/lib/sse-utils.ts`

The frontend goal is:

- update screens when another org member changes shared data
- avoid stale workflow/member/invitation views
- support one editor at a time for workflow canvas
- recover cleanly after reconnects or org switching

## 1. Core frontend pattern

Use one SSE connection per signed-in user for the currently active organization.

Do not open:

- one SSE stream per workflow
- one stream per members page
- one stream per tab section

Instead:

- mount one org-scoped collaboration stream high in the app
- route incoming events to the appropriate cache/store refresh handlers

This matches the current architecture better and keeps connection count under control.

## 2. Mount point

Create a global collaboration provider or effect that mounts once after auth and org initialization.

Suggested files:

- `src/lib/org-collaboration.ts`
- `src/components/general/OrgCollaborationProvider.tsx`

Mount it near your authenticated app shell, after:

- Clerk auth is ready
- `useOrganizationStore().currentOrganization` is known

The provider should reconnect whenever:

- auth token changes
- current organization changes
- browser comes back online

## 3. SSE connection strategy

Recommended endpoint:

- `GET /api/collaboration/events`

Use the active org already carried in the JWT/backend middleware. No org ID query param is needed.

### Transport choice

Use native `EventSource` if your backend can authenticate with cookies.

If you need Bearer token auth, use `fetch` plus the existing `readSSEStream()` helper in `src/lib/sse-utils.ts`, because native `EventSource` does not let you attach custom auth headers.

For this codebase, the safer default is a `fetch`-based SSE client because `backendApiClient` already depends on Clerk token retrieval.

## 4. Suggested client module

Create a small collaboration client module:

```ts
export type OrgCollaborationEvent =
  | { event_type: 'workflow.created'; organization_id: number; actor_user_id: number; payload: { workflow_id: string; name?: string } }
  | { event_type: 'workflow.updated'; organization_id: number; actor_user_id: number; payload: { workflow_id: string; changed_fields?: string[] } }
  | { event_type: 'workflow.draft.updated'; organization_id: number; actor_user_id: number; payload: { workflow_id: string } }
  | { event_type: 'workflow.deleted'; organization_id: number; actor_user_id: number; payload: { workflow_id: string } }
  | { event_type: 'organization.updated'; organization_id: number; actor_user_id: number; payload: {} }
  | { event_type: 'organization.member.added'; organization_id: number; actor_user_id: number; payload: { user_id: number } }
  | { event_type: 'organization.member.removed'; organization_id: number; actor_user_id: number; payload: { user_id: number } }
  | { event_type: 'organization.member.role.updated'; organization_id: number; actor_user_id: number; payload: { user_id: number; role: string } }
  | { event_type: 'organization.invitation.created'; organization_id: number; actor_user_id: number; payload: { invitation_id: number } }
  | { event_type: 'organization.invitation.accepted'; organization_id: number; actor_user_id: number; payload: { user_id: number } }
  | { event_type: 'workflow.lock.acquired'; organization_id: number; actor_user_id: number; payload: { workflow_id: string; holder_user_id: number; holder_name?: string; tab_id?: string } }
  | { event_type: 'workflow.lock.released'; organization_id: number; actor_user_id: number; payload: { workflow_id: string } };
```

Keep this typed. Do not leave the event bus as unstructured `any`.

## 5. Self-event policy

Ignore most events produced by the current user.

Reason:

- local mutation code already updates React Query and Zustand immediately
- processing your own invalidation event often causes redundant refetches and UI jitter

Rule:

- if `event.actor_user_id === currentUserId`, ignore

Exceptions:

- lock events if multiple tabs from the same user matter
- delete events if the local tab did not perform the delete

## 6. Reconnect behavior

The provider must handle disconnects as normal behavior.

Recommended behavior:

1. reconnect with backoff
2. on successful reconnect, invalidate a small set of org-wide queries once
3. if a workflow is currently open, refetch its detail
4. if members UI is open, refetch members and invitations

Initial backoff:

- `1s`
- `2s`
- `5s`
- `10s`
- cap at `30s`

Do not rely only on replay. Also do one defensive refresh after reconnect.

## 7. Event handling matrix

### Workflow events

For these events:

- `workflow.created`
- `workflow.updated`
- `workflow.draft.updated`
- `workflow.deleted`
- `workflow.published`
- `workflow.unpublished`
- `workflow.active.changed`
- `workflow.version.restored`
- `workflow.transferred`

Do this:

- `queryClient.invalidateQueries({ queryKey: workflowKeys.list() })`
- if `payload.workflow_id` exists:
  - `queryClient.invalidateQueries({ queryKey: workflowKeys.detail(payload.workflow_id) })`
  - `queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(payload.workflow_id) })`

Special delete handling:

- if deleted workflow is selected in `workflowStore`, clear selection and route away if needed
- remove local cache entry if you already know it is deleted

### Organization events

For:

- `organization.updated`

Do this:

- `useOrganizationStore.getState().fetchOrganizations()`

Why:

- org data is in Zustand, not only in React Query

### Member events

For:

- `organization.member.added`
- `organization.member.removed`
- `organization.member.role.updated`

Do this:

- `useMembersStore.getState().fetchMembers()`
- `useOrganizationStore.getState().fetchOrganizations()` if role changes can affect permissions displayed in header/settings

### Invitation events

For:

- `organization.invitation.created`
- `organization.invitation.revoked`
- `organization.invitation.accepted`

Do this:

- `useMembersStore.getState().fetchInvitations()`
- for accepted invites also call `fetchMembers()`

### Approval and integration events

If you add those event types later:

- invalidate approval-related views
- refresh shared integrations list

## 8. Central invalidation handler

Create one event handler function instead of spreading logic across random components.

Suggested file:

- `src/lib/org-collaboration.ts`

Pseudo-implementation:

```ts
import { queryClient } from '@/lib/query-client';
import { workflowKeys } from '@/lib/query-keys';
import { useMembersStore } from '@/stores/membersStore';
import { useOrganizationStore } from '@/stores/organizationStore';
import { useWorkflowStore } from '@/stores/workflowStore';

export async function handleOrgCollaborationEvent(event: OrgCollaborationEvent) {
  switch (event.event_type) {
    case 'workflow.created':
    case 'workflow.updated':
    case 'workflow.draft.updated':
    case 'workflow.published':
    case 'workflow.unpublished':
    case 'workflow.active.changed':
    case 'workflow.version.restored':
    case 'workflow.transferred':
      await queryClient.invalidateQueries({ queryKey: workflowKeys.list() });
      if (event.payload.workflow_id) {
        await Promise.allSettled([
          queryClient.invalidateQueries({ queryKey: workflowKeys.detail(event.payload.workflow_id) }),
          queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(event.payload.workflow_id) }),
        ]);
      }
      return;

    case 'workflow.deleted':
      await queryClient.invalidateQueries({ queryKey: workflowKeys.list() });
      if (event.payload.workflow_id) {
        queryClient.removeQueries({ queryKey: workflowKeys.detail(event.payload.workflow_id) });
        queryClient.removeQueries({ queryKey: workflowKeys.versionList(event.payload.workflow_id) });
      }
      return;

    case 'organization.updated':
      await useOrganizationStore.getState().fetchOrganizations();
      return;

    case 'organization.member.added':
    case 'organization.member.removed':
    case 'organization.member.role.updated':
      await useMembersStore.getState().fetchMembers();
      return;

    case 'organization.invitation.created':
    case 'organization.invitation.revoked':
      await useMembersStore.getState().fetchInvitations();
      return;

    case 'organization.invitation.accepted':
      await Promise.all([
        useMembersStore.getState().fetchMembers(),
        useMembersStore.getState().fetchInvitations(),
      ]);
      return;
  }
}
```

## 9. Workflow canvas lock state

Add a dedicated Zustand store for collaboration lock state.

Suggested file:

- `src/stores/workflowCollaborationStore.ts`

Suggested state:

```ts
interface WorkflowLockState {
  locksByWorkflowId: Record<string, {
    holderUserId: number;
    holderName?: string;
    acquiredAt?: string;
    expiresAt?: string;
    tabId?: string;
  } | null>;
  activeLockWorkflowId: string | null;
  lockStatus: 'idle' | 'acquiring' | 'held' | 'conflicted';
  lastLockError: string | null;
}
```

The collaboration provider updates this store from `workflow.lock.*` events.

## 10. Lock acquisition UX

Recommended UX:

- read-only by default is not necessary
- acquire lock when user enters edit mode, or on first edit attempt
- if lock acquired, enable editing and start heartbeat timer
- if lock denied, keep canvas read-only and show holder banner

Banner examples:

- `Jane is editing this workflow`
- `You are editing this workflow`
- `Editing lock lost. Canvas is now read-only.`

## 11. Lock-aware workflow page behavior

On the workflow page:

- if current workflow is open and another member acquires lock, set canvas `readOnly`
- if current user holds lock, keep normal editing
- if lock released and user has permission, allow retry acquire

Apply this specifically around:

- `WorkflowCanvas`
- save draft actions
- publish action
- rename/update metadata actions if they mutate shared workflow state

## 12. Lock heartbeat lifecycle

When current tab holds the lock:

- start heartbeat every `10s`
- stop heartbeat on:
  - route leave
  - tab close
  - page hidden for too long
  - org switch
  - workflow switch

Try to release on unload, but do not rely on it. TTL expiration is the real cleanup mechanism.

## 13. Lock conflict behavior

If acquire returns `409`:

- store current holder metadata
- show non-dismissible read-only warning while lock exists
- offer retry button

Do not silently downgrade to editing anyway. That defeats the whole point.

## 14. Org switching behavior

When `organizationStore.switchOrganization()` succeeds:

- close current collaboration stream
- clear workflow lock state
- stop any active heartbeat
- remove workflow queries as you already do
- reconnect stream for the new active org only after token refresh completes

This is important because org context is derived from the new JWT.

## 15. Multi-tab behavior

Decide this explicitly.

Recommended first cut:

- lock is per browser tab, not per user
- generate a `tabId` once per tab and include it in lock acquire/heartbeat/release

Why:

- it avoids two tabs from the same user editing independently without warning

Frontend rules:

- if event is from same user but different `tabId`, do not ignore lock events
- show `You are editing this in another tab` if useful

## 16. Minimal UI surfaces to update

You do not need to touch every component directly. The important surfaces are:

- workflow list page
- current workflow detail/canvas page
- settings team members section
- invitations section
- approval list section if present

Everything else should update via shared store/query invalidation.

## 17. Suggested provider shape

Example structure:

```tsx
export function OrgCollaborationProvider({ children }: { children: React.ReactNode }) {
  const { isLoaded, isSignedIn, userId } = useAuth();
  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);

  useEffect(() => {
    if (!isLoaded || !isSignedIn || !currentOrganization) return;

    let cancelled = false;
    let abortController: AbortController | null = null;

    async function connect() {
      abortController = new AbortController();
      // fetch SSE stream with Clerk token
      // parse via readSSEStream()
      // dispatch events to central handler
    }

    void connect();
    return () => {
      cancelled = true;
      abortController?.abort();
    };
  }, [isLoaded, isSignedIn, currentOrganization?.id, userId]);

  return <>{children}</>;
}
```

Keep the connection management in one place.

## 18. Defensive refresh on reconnect

When the connection transitions from disconnected to connected:

- `invalidateQueries({ queryKey: workflowKeys.list() })`
- if a workflow is selected:
  - `invalidateQueries({ queryKey: workflowKeys.detail(selectedWorkflowId) })`
  - `invalidateQueries({ queryKey: workflowKeys.versionList(selectedWorkflowId) })`
- if team settings UI is open:
  - `fetchMembers()`
  - `fetchInvitations()`

This protects you against missed gaps and expired stream history.

## 19. Toasts vs silent refresh

Most invalidation events should be silent.

Do show UI messages for:

- lock acquired by another user while I am viewing the workflow
- lock lost while I am editing
- workflow deleted while I am on that workflow
- workflow changed by another user while I have unsaved local edits

That last case matters.

## 20. Unsaved local edits and remote changes

For workflow canvas, invalidation alone is not enough if the current user has unsaved edits.

Recommended rule:

- if remote `workflow.draft.updated` arrives for the currently open workflow
- and local canvas is dirty
- and current tab does not hold the lock

Then:

- keep current local state untouched
- show banner: `This workflow changed remotely. Refresh to load latest version.`
- provide `Reload` action

If you later enforce locking strictly on all draft mutations, this case should be rare, but the UI still needs to handle it.

## 21. Testing plan

### Unit tests

Add tests for:

- event parsing into typed collaboration events
- invalidation handler calls correct query/store refreshes
- self-event suppression
- lock event handling updates lock store correctly

### Component tests

Add tests for:

- provider reconnect triggers defensive refresh
- workflow page becomes read-only on remote lock acquired
- members screen refreshes on member events
- invitations screen refreshes on invitation events

### E2E

Add a multi-context Playwright test:

1. user A and user B open same org
2. user A edits workflow name or draft
3. user B workflow list/detail refreshes
4. user A acquires canvas lock
5. user B sees read-only indicator
6. user A closes tab or stops heartbeat
7. user B can acquire lock after TTL

## 22. Recommended first cut for this frontend

Implement this order:

1. org collaboration provider with one SSE connection
2. central invalidation handler for workflows, members, invitations, organization
3. reconnect backoff plus defensive refresh
4. workflow lock store and read-only banner
5. lock acquire/heartbeat/release flow on workflow page

That order gets stale-screen prevention in place first, then adds editor coordination second.

## 23. Practical rule for this codebase

Use React Query invalidation for shared fetched data.

Use Zustand refresh/update for:

- `organizationStore`
- `membersStore`
- dedicated workflow lock state

Do not try to turn every collaboration event into an in-place manual cache patch. For your current app shape, invalidation plus selective refetch is simpler and less error-prone.
