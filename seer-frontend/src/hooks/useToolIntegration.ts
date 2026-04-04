/**
 * useToolIntegration Hook
 *
 * Phase 2: Refactored to use store directly instead of wrapper hook.
 * Provides helper logic for individual tool integration status and authentication.
 *
 * Enhanced to support connection-specific status queries for multi-account scenarios.
 * When connectionId is provided, fetches status for that specific OAuth connection.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useCurrentUser } from '@/hooks/useAuthProvider';
import { getToolStatusForConnection, type SingleToolStatusResponse } from '@/lib/api-client';
import type { ToolIntegrationStatus } from '@/stores/toolsStore';
import { useConnectIntegration } from './useConnectIntegration';
import { useToolingData } from './useToolingData';

export interface UseToolIntegrationResult {
  status: ToolIntegrationStatus | null;
  isLoading: boolean;
  userEmail: string | null;
  initiateAuth: () => Promise<string | null>;
  refresh: () => Promise<void>;
}

export function useToolIntegration(
  toolName: string,
  connectionId?: number | null,
): UseToolIntegrationResult {
  const { user } = useCurrentUser();
  const userEmail = user?.primaryEmailAddress?.emailAddress ?? user?.emailAddresses?.[0]?.emailAddress ?? null;

  // FIXED: Individual selectors instead of useShallow
  const connectIntegration = useConnectIntegration();
  const { toolsLoading: isLoading, getToolIntegrationStatus, refreshToolingData: refresh } = useToolingData();

  // Base status from global store
  const baseStatus = useMemo(() => getToolIntegrationStatus(toolName), [getToolIntegrationStatus, toolName]);

  // Connection-specific status when connectionId is provided
  const [connectionStatus, setConnectionStatus] = useState<SingleToolStatusResponse | null>(null);
  const [connectionStatusLoading, setConnectionStatusLoading] = useState(false);

  // Fetch connection-specific status when connectionId is provided
  useEffect(() => {
    if (!toolName || connectionId == null) {
      setConnectionStatus(null);
      return;
    }

    let cancelled = false;
    setConnectionStatusLoading(true);

    getToolStatusForConnection(toolName, connectionId)
      .then((response) => {
        if (!cancelled) {
          setConnectionStatus(response);
        }
      })
      .catch((error) => {
        console.warn('[useToolIntegration] Failed to fetch connection-specific status:', error);
        if (!cancelled) {
          setConnectionStatus(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setConnectionStatusLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [toolName, connectionId]);

  // Merge base status with connection-specific status
  const status = useMemo<ToolIntegrationStatus | null>(() => {
    if (!baseStatus) return null;

    // If we have connection-specific status, override the connected state
    if (connectionId != null && connectionStatus) {
      return {
        ...baseStatus,
        isConnected: connectionStatus.connected,
        connectionId: connectionStatus.connection_id,
      };
    }

    return baseStatus;
  }, [baseStatus, connectionId, connectionStatus]);

  const initiateAuth = useCallback(async () => {
    if (!status?.integrationType) {
      return null;
    }
    return connectIntegration(status.integrationType, { toolNames: [status.tool.name] });
  }, [status, connectIntegration]);

  // Combined loading state
  const combinedLoading = isLoading || connectionStatusLoading;

  return { status, isLoading: combinedLoading, userEmail, initiateAuth, refresh };
}
