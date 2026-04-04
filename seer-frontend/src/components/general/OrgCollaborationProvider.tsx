import { useAuthStatus } from '@/hooks/useAuthProvider';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import {
  getOrgCollaborationTabId,
  handleOrgCollaborationEvent,
  isOrgCollaborationEvent,
  refreshOrgCollaborationData,
  shouldIgnoreOrgEvent,
  type OrgCollaborationEvent,
} from '@/lib/org-collaboration';
import { backendApiClient } from '@/lib/api-client';
import { readSSEStream } from '@/lib/sse-utils';
import { useOrganizationStore, useWorkflowStore } from '@/stores';
import { useWorkflowCollaborationStore } from '@/stores/workflowCollaborationStore';

const RECONNECT_DELAYS_MS = [1000, 2000, 5000, 10000, 30000] as const;

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

function getRouteWorkflowId(pathname: string): string | null {
  const match = pathname.match(/^\/workflows\/([^/]+)$/);
  return match?.[1] ?? null;
}

async function startOrgCollaborationStream(params: {
  getLocationPathname: () => string;
  navigate: ReturnType<typeof useNavigate>;
  onReconnect: () => Promise<void> | void;
  tabId: string;
  userId: string | null | undefined;
  cancelledRef: { current: boolean };
  abortControllerRef: { current: AbortController | null };
  lastEventIdRef: { current: string | null };
  hasConnectedRef: { current: boolean };
}) {
  let reconnectAttempt = 0;

  while (!params.cancelledRef.current) {
    params.abortControllerRef.current = new AbortController();

    try {
      // Include tab_id so the backend can associate locks with this SSE connection
      const url = `/api/collaboration/events?tab_id=${encodeURIComponent(params.tabId)}`;
      const response = await backendApiClient.stream(url, {
        signal: params.abortControllerRef.current.signal,
        headers: params.lastEventIdRef.current
          ? { 'Last-Event-ID': params.lastEventIdRef.current }
          : undefined,
      });

      if (params.cancelledRef.current) {
        return;
      }

      const isReconnect = params.hasConnectedRef.current;
      params.hasConnectedRef.current = true;
      reconnectAttempt = 0;

      if (isReconnect) {
        await params.onReconnect();
      }

      await processOrgSseResponse(response, params);
    } catch (error) {
      if (
        params.cancelledRef.current ||
        params.abortControllerRef.current?.signal.aborted
      ) {
        return;
      }

      console.warn('[OrgCollaborationProvider] collaboration stream disconnected', error);
    }

    if (params.cancelledRef.current) {
      return;
    }

    const delay =
      RECONNECT_DELAYS_MS[Math.min(reconnectAttempt, RECONNECT_DELAYS_MS.length - 1)];
    reconnectAttempt += 1;
    await wait(delay);
  }
}

async function processOrgSseResponse(
  response: Response,
  params: Parameters<typeof startOrgCollaborationStream>[0],
) {
  for await (const sseEvent of readSSEStream<OrgCollaborationEvent>(response)) {
    if (params.cancelledRef.current) {
      break;
    }

    if (sseEvent.id && sseEvent.id !== 'sync-required') {
      params.lastEventIdRef.current = sseEvent.id;
    }

    if (!isOrgCollaborationEvent(sseEvent.data)) {
      continue;
    }

    if (shouldIgnoreOrgEvent(sseEvent.data, params.userId, params.tabId)) {
      continue;
    }

    await handleOrgCollaborationEvent(sseEvent.data, {
      currentClerkUserId: params.userId,
      currentTabId: params.tabId,
      onWorkflowDeleted: (workflowId) => {
        if (getRouteWorkflowId(params.getLocationPathname()) === workflowId) {
          params.navigate('/workflows', { replace: true });
        }
      },
    });
  }
}

export function OrgCollaborationProvider() {
  const { isLoaded, isSignedIn, userId } = useAuthStatus();
  const currentOrganization = useOrganizationStore((state) => state.currentOrganization);
  const selectedWorkflowId = useWorkflowStore((state) => state.selectedWorkflowId);
  const clearCollaborationState = useWorkflowCollaborationStore((state) => state.clearAll);
  const bumpReconnectRevision = useWorkflowCollaborationStore((state) => state.bumpReconnectRevision);

  const navigate = useNavigate();
  const location = useLocation();
  const [onlineRevision, setOnlineRevision] = useState(0);
  const tabId = useMemo(() => getOrgCollaborationTabId(), []);
  const lastEventIdRef = useRef<string | null>(null);
  const hasConnectedRef = useRef(false);

  const routeWorkflowId = useMemo(
    () => getRouteWorkflowId(location.pathname),
    [location.pathname],
  );
  const activeWorkflowId = routeWorkflowId ?? selectedWorkflowId;
  const activeWorkflowIdRef = useRef<string | null>(activeWorkflowId);
  const locationPathnameRef = useRef(location.pathname);
  const currentOrganizationId = currentOrganization?.id ?? null;

  useEffect(() => {
    activeWorkflowIdRef.current = activeWorkflowId;
  }, [activeWorkflowId]);

  useEffect(() => {
    locationPathnameRef.current = location.pathname;
  }, [location.pathname]);

  useEffect(() => {
    const handleOnline = () => {
      setOnlineRevision((value) => value + 1);
    };

    window.addEventListener('online', handleOnline);
    return () => window.removeEventListener('online', handleOnline);
  }, []);

  useEffect(() => {
    lastEventIdRef.current = null;
    hasConnectedRef.current = false;
    clearCollaborationState();
  }, [clearCollaborationState, currentOrganizationId]);

  useEffect(() => {
    if (isSignedIn) {
      return;
    }

    lastEventIdRef.current = null;
    hasConnectedRef.current = false;
    clearCollaborationState();
  }, [clearCollaborationState, isSignedIn]);

  useEffect(() => {
    if (!isLoaded || !isSignedIn || !currentOrganizationId) {
      return;
    }

    const cancelledRef = { current: false };
    const abortControllerRef = { current: null as AbortController | null };

    void startOrgCollaborationStream({
      getLocationPathname: () => locationPathnameRef.current,
      navigate,
      onReconnect: async () => {
        bumpReconnectRevision();
        await refreshOrgCollaborationData(activeWorkflowIdRef.current);
      },
      tabId,
      userId,
      cancelledRef,
      abortControllerRef,
      lastEventIdRef,
      hasConnectedRef,
    });

    return () => {
      cancelledRef.current = true;
      abortControllerRef.current?.abort();
    };
  }, [
    bumpReconnectRevision,
    currentOrganizationId,
    isLoaded,
    isSignedIn,
    navigate,
    onlineRevision,
    tabId,
    userId,
  ]);

  return null;
}
