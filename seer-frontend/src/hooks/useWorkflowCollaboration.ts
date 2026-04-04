import { useAuthStatus } from '@/hooks/useAuthProvider';
import { useCallback, useEffect, useMemo, useRef } from 'react';

import {
  acquireWorkflowLock,
  getOrgCollaborationTabId,
  getWorkflowLock,
  isWorkflowLockConflict,
  releaseWorkflowLock,
  type WorkflowLockMetadata,
  type WorkflowLockResponse,
} from '@/lib/org-collaboration';
import { useWorkflowCollaborationStore } from '@/stores/workflowCollaborationStore';

function getConflictLock(error: unknown): WorkflowLockMetadata | null {
  if (!isWorkflowLockConflict(error)) {
    return null;
  }

  const response = error.response as Partial<WorkflowLockResponse> | undefined;
  return response?.lock ?? null;
}

function isLockHeldByTab(
  lock: WorkflowLockMetadata | null | undefined,
  userId: string | null | undefined,
  tabId: string,
) {
  return Boolean(
    lock &&
      lock.holder_clerk_user_id === userId &&
      lock.tab_id === tabId,
  );
}

export function useWorkflowCollaboration(workflowId: string | null | undefined) {
  const { isLoaded, isSignedIn, userId } = useAuthStatus();
  const tabId = useMemo(() => getOrgCollaborationTabId(), []);
  const isAcquiringRef = useRef(false);

  const currentLock = useWorkflowCollaborationStore(
    useCallback(
      (state) => (workflowId ? state.locksByWorkflowId[workflowId] ?? null : null),
      [workflowId],
    ),
  );
  const lockStatus = useWorkflowCollaborationStore((state) => state.lockStatus);
  const lastLockError = useWorkflowCollaborationStore((state) => state.lastLockError);
  const reconnectRevision = useWorkflowCollaborationStore((state) => state.reconnectRevision);

  const pendingLockWorkflowId = useWorkflowCollaborationStore(
    (state) => state.pendingLockWorkflowId,
  );
  const setCurrentWorkflowId = useWorkflowCollaborationStore((state) => state.setCurrentWorkflowId);
  const beginLockAcquisition = useWorkflowCollaborationStore((state) => state.beginLockAcquisition);
  const markLockHeld = useWorkflowCollaborationStore((state) => state.markLockHeld);
  const markLockConflict = useWorkflowCollaborationStore((state) => state.markLockConflict);
  const clearWorkflowLock = useWorkflowCollaborationStore((state) => state.clearWorkflowLock);
  const setLastLockError = useWorkflowCollaborationStore((state) => state.setLastLockError);
  const setPendingLockWorkflowId = useWorkflowCollaborationStore(
    (state) => state.setPendingLockWorkflowId,
  );

  const isHeldByCurrentTab = isLockHeldByTab(currentLock, userId, tabId);
  const isHeldByOtherTab =
    Boolean(
      currentLock &&
        currentLock.holder_clerk_user_id === userId &&
        currentLock.tab_id &&
        currentLock.tab_id !== tabId,
    );
  const isReadOnly = Boolean(workflowId) && !isHeldByCurrentTab;

  const refreshLockState = useCallback(async () => {
    if (!workflowId || !isLoaded || !isSignedIn) {
      return;
    }

    try {
      const lock = await getWorkflowLock(workflowId);
      if (!lock) {
        clearWorkflowLock(workflowId);
        return;
      }

      if (lock.holder_clerk_user_id === userId && lock.tab_id === tabId) {
        markLockHeld(lock);
        return;
      }

      markLockConflict(workflowId, lock, null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to refresh workflow lock';
      setLastLockError(message);
    }
  }, [
    clearWorkflowLock,
    isLoaded,
    isSignedIn,
    markLockConflict,
    markLockHeld,
    setLastLockError,
    tabId,
    userId,
    workflowId,
  ]);

  const acquireLock = useCallback(async () => {
    if (!workflowId || !isLoaded || !isSignedIn) {
      return;
    }

    // Prevent duplicate acquisition calls (e.g., from React Strict Mode double-mount)
    if (isAcquiringRef.current) {
      return;
    }

    const existingLock =
      useWorkflowCollaborationStore.getState().locksByWorkflowId[workflowId] ?? null;
    if (isLockHeldByTab(existingLock, userId, tabId)) {
      return;
    }

    isAcquiringRef.current = true;
    beginLockAcquisition(workflowId);

    try {
      const lock = await acquireWorkflowLock(workflowId, tabId);
      markLockHeld(lock);
      setPendingLockWorkflowId(workflowId);
    } catch (error) {
      const conflictLock = getConflictLock(error);
      if (conflictLock) {
        // If the lock is held by the same user (stale tab/session),
        // force-release and re-acquire instead of going read-only
        if (conflictLock.holder_clerk_user_id === userId && conflictLock.tab_id) {
          try {
            await releaseWorkflowLock(workflowId, conflictLock.tab_id);
            const lock = await acquireWorkflowLock(workflowId, tabId);
            markLockHeld(lock);
            setPendingLockWorkflowId(workflowId);
            return;
          } catch {
            // Fall through to conflict state if force-release fails
          }
        }
        markLockConflict(workflowId, conflictLock, null);
        return;
      }

      const message = error instanceof Error ? error.message : 'Failed to acquire workflow lock';
      markLockConflict(workflowId, existingLock, message);
    } finally {
      isAcquiringRef.current = false;
    }
  }, [
    beginLockAcquisition,
    isLoaded,
    isSignedIn,
    markLockConflict,
    markLockHeld,
    setPendingLockWorkflowId,
    tabId,
    userId,
    workflowId,
  ]);

  const releaseLock = useCallback(async () => {
    if (!workflowId) {
      return;
    }

    const existingLock =
      useWorkflowCollaborationStore.getState().locksByWorkflowId[workflowId] ?? null;
    if (!isLockHeldByTab(existingLock, userId, tabId)) {
      return;
    }

    try {
      await releaseWorkflowLock(workflowId, tabId);
    } catch {
      // TTL cleanup is the fallback if the request does not complete.
    } finally {
      clearWorkflowLock(workflowId);
    }
  }, [clearWorkflowLock, tabId, userId, workflowId]);

  useEffect(() => {
    setCurrentWorkflowId(workflowId ?? null);

    return () => {
      setCurrentWorkflowId(null);
      // Clear pending lock when navigating away from this workflow
      setPendingLockWorkflowId(null);
    };
  }, [setCurrentWorkflowId, setPendingLockWorkflowId, workflowId]);

  useEffect(() => {
    if (!workflowId || !isLoaded || !isSignedIn) {
      return;
    }

    void acquireLock();

    return () => {
      void releaseLock();
    };
  }, [acquireLock, isLoaded, isSignedIn, releaseLock, workflowId]);

  // Lock renewal is now handled automatically via SSE connection lifecycle.
  // When the SSE connection is alive, the lock remains valid.
  // When the SSE connection disconnects, the backend automatically releases the lock.

  // Auto-reacquire lock after SSE reconnection if this tab was previously editing
  useEffect(() => {
    if (!workflowId || reconnectRevision === 0) {
      return;
    }

    const attemptReacquisition = async () => {
      // Check if this tab was previously editing this workflow
      const wasEditing = pendingLockWorkflowId === workflowId;

      if (!wasEditing) {
        // Not previously editing, just refresh the lock state
        void refreshLockState();
        return;
      }

      // Wait briefly to avoid race conditions with rapid reconnects
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Check current lock state from server
      try {
        const lock = await getWorkflowLock(workflowId);

        if (!lock) {
          // No lock exists, auto-reacquire since we were editing
          void acquireLock();
        } else if (lock.holder_clerk_user_id === userId && lock.tab_id === tabId) {
          // We still hold the lock (shouldn't happen but handle gracefully)
          markLockHeld(lock);
        } else {
          // Someone else has the lock now, show conflict
          markLockConflict(workflowId, lock, null);
        }
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'Failed to check workflow lock';
        setLastLockError(message);
      }
    };

    void attemptReacquisition();
  }, [
    acquireLock,
    markLockConflict,
    markLockHeld,
    pendingLockWorkflowId,
    reconnectRevision,
    refreshLockState,
    setLastLockError,
    tabId,
    userId,
    workflowId,
  ]);

  const lockMessage = useMemo(() => {
    if (lockStatus === 'acquiring') {
      return 'Checking edit lock…';
    }

    if (isHeldByOtherTab) {
      return 'You are editing this workflow in another tab.';
    }

    if (currentLock && !isHeldByCurrentTab) {
      return currentLock.holder_name
        ? `${currentLock.holder_name} is editing this workflow.`
        : 'Another member is editing this workflow.';
    }

    return lastLockError;
  }, [currentLock, isHeldByCurrentTab, isHeldByOtherTab, lastLockError, lockStatus]);

  return {
    tabId,
    lock: currentLock,
    lockStatus,
    isReadOnly,
    isHeldByCurrentTab,
    lockMessage,
    acquireLock,
    refreshLockState,
    releaseLock,
  };
}
