import { BackendAPIError, backendApiClient } from '@/lib/api-client';
import { queryClient } from '@/lib/query-client';
import { workflowKeys } from '@/lib/query-keys';
import { useCanvasStore, useChatStore, useMembersStore, useOrganizationStore, useTriggersStore, useUIStore, useWorkflowStore } from '@/stores';
import { useWorkflowCollaborationStore } from '@/stores/workflowCollaborationStore';

export type OrgCollaborationEventType =
  | 'workflow.created'
  | 'workflow.updated'
  | 'workflow.draft.updated'
  | 'workflow.published'
  | 'workflow.unpublished'
  | 'workflow.version.restored'
  | 'workflow.deleted'
  | 'workflow.transferred'
  | 'organization.updated'
  | 'organization.member.added'
  | 'organization.member.removed'
  | 'organization.member.role.updated'
  | 'organization.invitation.created'
  | 'organization.invitation.accepted'
  | 'organization.invitation.revoked'
  | 'workflow.lock.acquired'
  | 'workflow.lock.released'
  | 'workflow.lock.expired'
  | 'sync.required';

export interface WorkflowLockMetadata {
  organization_id: number;
  workflow_id: string;
  holder_clerk_user_id: string | null;
  holder_db_user_id: number | null;
  holder_name: string | null;
  acquired_at?: string;
  expires_at?: string;
  tab_id: string | null;
  sse_connection_id?: string | null;
}

export interface WorkflowLockResponse {
  lock: WorkflowLockMetadata;
}

export interface WorkflowLockStatusResponse {
  lock: WorkflowLockMetadata | null;
}

type CollaborationEventPayload = Record<string, unknown> & {
  workflow_id?: string;
  holder_name?: string;
  tab_id?: string;
};

export interface OrgCollaborationEvent {
  event_type: OrgCollaborationEventType;
  organization_id: number;
  actor_clerk_user_id?: string | null;
  actor_db_user_id?: number | null;
  resource_type?: string;
  resource_id?: string | null;
  occurred_at?: string;
  correlation_id?: string | null;
  payload: CollaborationEventPayload;
}

interface HandleOrgCollaborationEventOptions {
  currentClerkUserId?: string | null;
  currentTabId?: string;
  onWorkflowDeleted?: (workflowId: string) => void;
}

const WORKFLOW_EVENT_TYPES: ReadonlySet<OrgCollaborationEventType> = new Set([
  'workflow.created',
  'workflow.updated',
  'workflow.draft.updated',
  'workflow.published',
  'workflow.unpublished',
  'workflow.version.restored',
  'workflow.transferred',
]);

const LOCK_RELEASE_EVENT_TYPES: ReadonlySet<OrgCollaborationEventType> = new Set([
  'workflow.lock.released',
  'workflow.lock.expired',
]);

export function getOrgCollaborationTabId(): string {
  if (typeof window === 'undefined') {
    return 'server';
  }

  const existing = window.sessionStorage.getItem('seer-org-collab-tab-id');
  if (existing) {
    return existing;
  }

  const nextId =
    typeof window.crypto?.randomUUID === 'function'
      ? window.crypto.randomUUID()
      : `tab-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

  window.sessionStorage.setItem('seer-org-collab-tab-id', nextId);
  return nextId;
}

export function isOrgCollaborationEvent(value: unknown): value is OrgCollaborationEvent {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<OrgCollaborationEvent>;
  return typeof candidate.event_type === 'string' && typeof candidate.organization_id === 'number';
}

export function shouldIgnoreOrgEvent(
  event: OrgCollaborationEvent,
  currentClerkUserId?: string | null,
  currentTabId?: string,
): boolean {
  if (!currentClerkUserId || event.actor_clerk_user_id !== currentClerkUserId) {
    return false;
  }

  if (event.event_type.startsWith('workflow.lock.')) {
    const eventTabId =
      typeof event.payload.tab_id === 'string'
        ? event.payload.tab_id
        : null;

    return !eventTabId || !currentTabId || eventTabId === currentTabId;
  }

  return true;
}

function resetDeletedWorkflowState() {
  useCanvasStore.getState().reset();
  useTriggersStore.setState({ workflowTriggers: new Map() });
  useUIStore.getState().resetUIState();
  useChatStore.getState().resetChatState();
  useWorkflowStore.setState({
    selectedWorkflowId: null,
    workflowName: 'My Workflow',
    workflowInputData: {},
    isLoadingWorkflow: false,
  });
}

function toLockMetadataFromEvent(event: OrgCollaborationEvent): WorkflowLockMetadata | null {
  const workflowId = event.payload.workflow_id ?? event.resource_id ?? null;
  if (!workflowId) {
    return null;
  }

  return {
    organization_id: event.organization_id,
    workflow_id: workflowId,
    holder_clerk_user_id: event.actor_clerk_user_id ?? null,
    holder_db_user_id: event.actor_db_user_id ?? null,
    holder_name:
      typeof event.payload.holder_name === 'string' ? event.payload.holder_name : null,
    tab_id: typeof event.payload.tab_id === 'string' ? event.payload.tab_id : null,
    acquired_at: event.occurred_at,
  };
}

export async function refreshOrgCollaborationData(activeWorkflowId?: string | null) {
  await Promise.allSettled([
    queryClient.invalidateQueries({ queryKey: workflowKeys.list() }),
    useOrganizationStore.getState().fetchOrganizations(),
    useMembersStore.getState().fetchMembers(),
    useMembersStore.getState().fetchInvitations(),
    activeWorkflowId
      ? queryClient.invalidateQueries({ queryKey: workflowKeys.detail(activeWorkflowId) })
      : Promise.resolve(),
    activeWorkflowId
      ? queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(activeWorkflowId) })
      : Promise.resolve(),
  ]);
}

async function handleWorkflowMutationEvent(workflowId: string | null) {
  await queryClient.invalidateQueries({ queryKey: workflowKeys.list() });

  if (workflowId) {
    await Promise.allSettled([
      queryClient.invalidateQueries({ queryKey: workflowKeys.detail(workflowId) }),
      queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(workflowId) }),
    ]);
  }
}

async function handleWorkflowDeletedEvent(
  workflowId: string | null,
  options: HandleOrgCollaborationEventOptions,
) {
  await queryClient.invalidateQueries({ queryKey: workflowKeys.list() });

  if (!workflowId) {
    return;
  }

  queryClient.removeQueries({ queryKey: workflowKeys.detail(workflowId) });
  queryClient.removeQueries({ queryKey: workflowKeys.versionList(workflowId) });
  useWorkflowCollaborationStore.getState().clearWorkflowLock(workflowId);

  const selectedWorkflowId = useWorkflowStore.getState().selectedWorkflowId;
  if (selectedWorkflowId === workflowId) {
    resetDeletedWorkflowState();
    options.onWorkflowDeleted?.(workflowId);
  }
}

async function handleOrganizationEvent(eventType: OrgCollaborationEventType) {
  if (eventType === 'organization.updated') {
    await useOrganizationStore.getState().fetchOrganizations();
    return;
  }

  if (
    eventType === 'organization.member.added' ||
    eventType === 'organization.member.removed' ||
    eventType === 'organization.member.role.updated'
  ) {
    await Promise.allSettled([
      useMembersStore.getState().fetchMembers(),
      useOrganizationStore.getState().fetchOrganizations(),
    ]);
    return;
  }

  if (
    eventType === 'organization.invitation.created' ||
    eventType === 'organization.invitation.revoked'
  ) {
    await useMembersStore.getState().fetchInvitations();
    return;
  }

  if (eventType === 'organization.invitation.accepted') {
    await Promise.allSettled([
      useMembersStore.getState().fetchMembers(),
      useMembersStore.getState().fetchInvitations(),
    ]);
  }
}

function handleWorkflowLockEvent(
  event: OrgCollaborationEvent,
  options: HandleOrgCollaborationEventOptions,
  workflowId: string | null,
) {
  const workflowCollaborationStore = useWorkflowCollaborationStore.getState();

  if (event.event_type === 'workflow.lock.acquired') {
    const lock = toLockMetadataFromEvent(event);
    if (!lock) {
      return;
    }

    workflowCollaborationStore.setWorkflowLock(lock.workflow_id, lock);
    const isCurrentTabHolder =
      Boolean(options.currentClerkUserId) &&
      lock.holder_clerk_user_id === options.currentClerkUserId &&
      Boolean(options.currentTabId) &&
      lock.tab_id === options.currentTabId;

    if (isCurrentTabHolder) {
      workflowCollaborationStore.markLockHeld(lock);
      return;
    }

    workflowCollaborationStore.markLockConflict(lock.workflow_id, lock, null);
    return;
  }

  if (workflowId && LOCK_RELEASE_EVENT_TYPES.has(event.event_type)) {
    workflowCollaborationStore.clearWorkflowLock(workflowId);
  }
}

export async function handleOrgCollaborationEvent(
  event: OrgCollaborationEvent,
  options: HandleOrgCollaborationEventOptions = {},
) {
  const workflowId = event.payload.workflow_id ?? event.resource_id ?? null;

  if (WORKFLOW_EVENT_TYPES.has(event.event_type)) {
    await handleWorkflowMutationEvent(workflowId);
    return;
  }

  if (event.event_type === 'workflow.deleted') {
    await handleWorkflowDeletedEvent(workflowId, options);
    return;
  }

  if (event.event_type.startsWith('organization.')) {
    await handleOrganizationEvent(event.event_type);
    return;
  }

  if (event.event_type.startsWith('workflow.lock.')) {
    handleWorkflowLockEvent(event, options, workflowId);
    return;
  }

  if (event.event_type === 'sync.required') {
    await refreshOrgCollaborationData(useWorkflowStore.getState().selectedWorkflowId);
  }
}

export async function getWorkflowLock(workflowId: string): Promise<WorkflowLockMetadata | null> {
  const response = await backendApiClient.request<WorkflowLockStatusResponse>(
    `/api/collaboration/workflows/${workflowId}/lock`,
  );
  return response.lock;
}

export async function acquireWorkflowLock(
  workflowId: string,
  tabId: string,
): Promise<WorkflowLockMetadata> {
  const response = await backendApiClient.request<WorkflowLockResponse>(
    `/api/collaboration/workflows/${workflowId}/lock`,
    {
      method: 'POST',
      body: { tab_id: tabId },
    },
  );
  return response.lock;
}

export async function releaseWorkflowLock(
  workflowId: string,
  tabId: string,
): Promise<void> {
  await backendApiClient.request(
    `/api/collaboration/workflows/${workflowId}/lock?tab_id=${encodeURIComponent(tabId)}`,
    { method: 'DELETE' },
  );
}

export function isWorkflowLockConflict(error: unknown): error is BackendAPIError {
  return error instanceof BackendAPIError && error.status === 409;
}
