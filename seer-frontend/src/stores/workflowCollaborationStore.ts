import { createStore } from './createStore';
import type { WorkflowLockMetadata } from '@/lib/org-collaboration';

export type WorkflowLockStatus = 'idle' | 'acquiring' | 'held' | 'conflicted';

export interface WorkflowLockState {
  locksByWorkflowId: Record<string, WorkflowLockMetadata | null>;
  activeLockWorkflowId: string | null;
  currentWorkflowId: string | null;
  lockStatus: WorkflowLockStatus;
  lastLockError: string | null;
  reconnectRevision: number;
  pendingLockWorkflowId: string | null; // Tracks "this tab was editing this workflow" for reconnection
  setCurrentWorkflowId: (workflowId: string | null) => void;
  setWorkflowLock: (workflowId: string, lock: WorkflowLockMetadata | null) => void;
  beginLockAcquisition: (workflowId: string) => void;
  markLockHeld: (lock: WorkflowLockMetadata) => void;
  markLockConflict: (
    workflowId: string,
    lock: WorkflowLockMetadata | null,
    errorMessage: string | null,
  ) => void;
  clearWorkflowLock: (workflowId: string) => void;
  setLastLockError: (message: string | null) => void;
  bumpReconnectRevision: () => void;
  setPendingLockWorkflowId: (workflowId: string | null) => void;
  clearAll: () => void;
}

const initialState = {
  locksByWorkflowId: {},
  activeLockWorkflowId: null,
  currentWorkflowId: null,
  lockStatus: 'idle' as WorkflowLockStatus,
  lastLockError: null,
  reconnectRevision: 0,
  pendingLockWorkflowId: null,
};

export const useWorkflowCollaborationStore = createStore<WorkflowLockState>((set, get) => ({
  ...initialState,

  setCurrentWorkflowId: (workflowId) => {
    set((state) => ({
      currentWorkflowId: workflowId,
      lockStatus:
        workflowId && state.activeLockWorkflowId === workflowId
          ? 'held'
          : workflowId && state.locksByWorkflowId[workflowId]
            ? 'conflicted'
            : 'idle',
      lastLockError: workflowId ? state.lastLockError : null,
    }));
  },

  setWorkflowLock: (workflowId, lock) => {
    set((state) => ({
      locksByWorkflowId: {
        ...state.locksByWorkflowId,
        [workflowId]: lock,
      },
    }));
  },

  beginLockAcquisition: (workflowId) => {
    set({
      currentWorkflowId: workflowId,
      lockStatus: 'acquiring',
      lastLockError: null,
    });
  },

  markLockHeld: (lock) => {
    set((state) => ({
      locksByWorkflowId: {
        ...state.locksByWorkflowId,
        [lock.workflow_id]: lock,
      },
      activeLockWorkflowId: lock.workflow_id,
      currentWorkflowId: lock.workflow_id,
      lockStatus: 'held',
      lastLockError: null,
    }));
  },

  markLockConflict: (workflowId, lock, errorMessage) => {
    set((state) => ({
      locksByWorkflowId: {
        ...state.locksByWorkflowId,
        [workflowId]: lock,
      },
      activeLockWorkflowId:
        state.activeLockWorkflowId === workflowId ? null : state.activeLockWorkflowId,
      lockStatus: state.currentWorkflowId === workflowId ? 'conflicted' : state.lockStatus,
      lastLockError: errorMessage,
    }));
  },

  clearWorkflowLock: (workflowId) => {
    set((state) => ({
      locksByWorkflowId: {
        ...state.locksByWorkflowId,
        [workflowId]: null,
      },
      activeLockWorkflowId:
        state.activeLockWorkflowId === workflowId ? null : state.activeLockWorkflowId,
      lockStatus:
        state.currentWorkflowId === workflowId && state.activeLockWorkflowId !== workflowId
          ? 'idle'
          : state.lockStatus,
      lastLockError:
        state.currentWorkflowId === workflowId ? null : state.lastLockError,
    }));
  },

  setLastLockError: (message) => {
    set({ lastLockError: message });
  },

  bumpReconnectRevision: () => {
    set((state) => ({ reconnectRevision: state.reconnectRevision + 1 }));
  },

  setPendingLockWorkflowId: (workflowId) => {
    set({ pendingLockWorkflowId: workflowId });
  },

  clearAll: () => {
    set(initialState);
  },
}));

export function getWorkflowLockState(workflowId: string | null | undefined) {
  if (!workflowId) {
    return null;
  }

  return useWorkflowCollaborationStore.getState().locksByWorkflowId[workflowId] ?? null;
}
