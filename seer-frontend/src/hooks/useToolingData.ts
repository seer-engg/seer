import { useMemo, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import type { ConnectedAccount } from '@/lib/api-client';
import type { IntegrationType } from '@/lib/integrations/client';
import { invalidateBuilderCatalogQueries } from '@/lib/query-client';
import { deriveIntegrationData, getIntegrationProviderForType, type ProviderConnectionInfo } from '@/stores/toolsStore.helpers';
import type { ToolIntegrationStatus, ToolMetadata, IntegrationGroup } from '@/stores/toolsStore';

import { useConnectionsQuery } from './useConnectionsQuery';
import { useToolCatalogQuery } from './useToolCatalogQuery';
import { useToolStatusQuery } from './useToolStatusQuery';

const EMPTY_TOOLS: ToolMetadata[] = [];
const EMPTY_CONNECTIONS: ConnectedAccount[] = [];

export function useRefreshToolingData() {
  const queryClient = useQueryClient();

  return useCallback(async () => invalidateBuilderCatalogQueries(queryClient), [queryClient]);
}

export function useToolingData() {
  const toolsQuery = useToolCatalogQuery();
  const toolStatusQuery = useToolStatusQuery();
  const connectionsQuery = useConnectionsQuery();
  const refreshToolingData = useRefreshToolingData();

  const tools = useMemo(() => toolsQuery.data ?? EMPTY_TOOLS, [toolsQuery.data]);
  const toolStatus = toolStatusQuery.data ?? null;
  const connections = useMemo(() => connectionsQuery.data ?? EMPTY_CONNECTIONS, [connectionsQuery.data]);

  const derived = useMemo(
    () => deriveIntegrationData({ tools, toolStatus, connections }),
    [tools, toolStatus, connections],
  );

  const getConnectionId = useCallback((type: IntegrationType) => {
    const connected = derived.connectedIntegrations.get(type);
    if (connected?.id) {
      return connected.id;
    }
    const provider = getIntegrationProviderForType(type, derived.integrationProviderMap);
    if (!provider) {
      return null;
    }
    return derived.providerConnectionsMap.get(provider)?.connectionId ?? null;
  }, [derived.connectedIntegrations, derived.integrationProviderMap, derived.providerConnectionsMap]);

  const isIntegrationConnected = useCallback(
    (type: IntegrationType) => derived.connectedIntegrations.has(type),
    [derived.connectedIntegrations],
  );

  const getToolIntegrationStatus = useCallback(
    (toolName: string): ToolIntegrationStatus | null =>
      derived.toolsWithStatus.find((status) => status.tool.name === toolName) ?? null,
    [derived.toolsWithStatus],
  );

  const getToolsByIntegration = useCallback(
    (integrationType: string): ToolMetadata[] => derived.toolsByIntegration.get(integrationType) ?? [],
    [derived.toolsByIntegration],
  );

  return {
    toolsQuery,
    toolStatusQuery,
    connectionsQuery,
    tools,
    toolStatus,
    connections,
    toolsLoading: toolsQuery.isPending || toolStatusQuery.isPending || connectionsQuery.isPending,
    toolsLoaded: toolsQuery.status === 'success',
    connectionsLoaded: connectionsQuery.status === 'success',
    toolsWithStatus: derived.toolsWithStatus,
    integrationProviderMap: derived.integrationProviderMap,
    integrationScopesMap: derived.integrationScopesMap,
    providerConnectionsMap: derived.providerConnectionsMap as Map<string, ProviderConnectionInfo>,
    toolStatusMap: derived.toolStatusMap,
    connectedIntegrations: derived.connectedIntegrations,
    toolsByIntegration: derived.toolsByIntegration,
    integrationGroups: derived.integrationGroups as IntegrationGroup[],
    getConnectionId,
    isIntegrationConnected,
    getToolIntegrationStatus,
    getToolsByIntegration,
    refreshToolingData,
  };
}
