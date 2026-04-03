import type { ConnectedAccount, ToolConnectionStatus } from '@/lib/api-client';
import { getOAuthProvider } from '@/lib/integrations/client';
import type { IntegrationType } from '@/lib/integrations/client';
import type { IntegrationGroup, ToolIntegrationStatus, ToolMetadata } from './toolsStore';
import { useIntegrationMetadataStore } from './integrationMetadataStore';

/**
 * Check if an integration requires OAuth connection.
 * Delegates to the metadata store for dynamic configuration.
 */
export const integrationNeedsConnection = (integrationType: IntegrationType | null): integrationType is IntegrationType => {
  if (!integrationType) return false;
  return useIntegrationMetadataStore.getState().requiresOAuth(integrationType);
};

/**
 * Get the integration type for a tool.
 * Uses explicit integration_type from tool metadata if available,
 * otherwise uses detection patterns from the metadata store.
 */
export const getIntegrationType = (tool: ToolMetadata): IntegrationType | null => {
  // If the tool has an explicit integration type, use it
  if (tool.integration_type) {
    return tool.integration_type as IntegrationType;
  }
  // Otherwise, detect from tool name and scopes using metadata store
  return useIntegrationMetadataStore.getState().detectIntegrationType(tool.name, tool.required_scopes);
};

/**
 * Get display name for an integration type.
 * Delegates to the metadata store for dynamic configuration.
 */
const getIntegrationDisplayName = (integrationType: string | null): string => {
  if (!integrationType) return 'Other';
  return useIntegrationMetadataStore.getState().getDisplayName(integrationType);
};

const buildProviderConnectionsMap = (connections: ConnectedAccount[]): Map<string, ProviderConnectionInfo> => {
  const map = new Map<string, ProviderConnectionInfo>();
  connections.forEach((connection) => {
    if (connection.status === 'ACTIVE') {
      const provider = connection.provider || connection.toolkit?.slug;
      if (provider) {
        const scopes = new Set<string>((connection.scopes || '').split(' ').filter(Boolean));
        map.set(provider, {
          scopes,
          connectionId: connection.id,
        });
      }
    }
  });
  return map;
};

const buildIntegrationProviderMap = (tools: ToolMetadata[]): Map<IntegrationType, string> => {
  const map = new Map<IntegrationType, string>();
  tools.forEach((tool) => {
    const integrationType = getIntegrationType(tool);
    if (integrationType && tool.provider) {
      map.set(integrationType, tool.provider);
    }
  });
  return map;
};

const buildIntegrationScopesMap = (tools: ToolMetadata[]): Map<IntegrationType, Set<string>> => {
  const map = new Map<IntegrationType, Set<string>>();
  tools.forEach((tool) => {
    const integrationType = getIntegrationType(tool);
    if (!integrationType) {
      return;
    }
    if (!map.has(integrationType)) {
      map.set(integrationType, new Set<string>());
    }
    const scopes = map.get(integrationType)!;
    tool.required_scopes.forEach((scope) => scopes.add(scope));
  });
  return map;
};

const buildToolStatusMap = (statuses: ToolConnectionStatus[] | null) => {
  if (!statuses) {
    return null;
  }
  const map = new Map<string, ToolConnectionStatus>();
  statuses.forEach((status) => {
    map.set(status.tool_name, status);
  });
  return map;
};

export const getIntegrationProviderForType = (
  integrationType: IntegrationType,
  integrationProviderMap: Map<IntegrationType, string>,
) => {
  return getOAuthProvider(integrationType) ?? integrationProviderMap.get(integrationType) ?? null;
};

export const getIntegrationScopesForType = (
  integrationType: IntegrationType,
  integrationScopesMap: Map<IntegrationType, Set<string>>,
) => {
  // First, check if we have scopes from tools
  const scopesFromTools = integrationScopesMap.get(integrationType);
  if (scopesFromTools && scopesFromTools.size > 0) {
    return Array.from(scopesFromTools);
  }
  // Fall back to default scopes from metadata store
  return useIntegrationMetadataStore.getState().getDefaultScopes(integrationType);
};

const computeToolsWithStatus = ({
  tools,
  toolStatusMap,
  providerConnectionsMap,
  integrationProviderMap,
  integrationScopesMap,
}: {
  tools: ToolMetadata[];
  toolStatusMap: Map<string, ToolConnectionStatus> | null;
  providerConnectionsMap: Map<string, ProviderConnectionInfo>;
  integrationProviderMap: Map<IntegrationType, string>;
  integrationScopesMap: Map<IntegrationType, Set<string>>;
}): ToolIntegrationStatus[] => {
  return tools.map((tool) => {
    const integrationType = getIntegrationType(tool);
    const needsConnection = integrationNeedsConnection(integrationType);
    if (!needsConnection) {
      return {
        tool,
        integrationType: null,
        isConnected: true,
        connectionId: null,
        requiredScopes: [],
      };
    }

    const assuredIntegrationType = integrationType!;
    const requiredScopes =
      tool.required_scopes.length > 0
        ? tool.required_scopes
        : getIntegrationScopesForType(assuredIntegrationType, integrationScopesMap);

    if (toolStatusMap) {
      const status = toolStatusMap.get(tool.name);
      if (status) {
        const isConnected = status.connected === true;
        return {
          tool,
          integrationType: assuredIntegrationType,
          isConnected,
          connectionId: status.connection_id,
          requiredScopes,
        };
      }
    }

    const provider = getIntegrationProviderForType(assuredIntegrationType, integrationProviderMap);
    if (provider) {
      const providerConn = providerConnectionsMap.get(provider);
      if (providerConn) {
        const hasAllScopes = requiredScopes.length === 0
          ? false
          : requiredScopes.every((scope) => providerConn.scopes.has(scope));
        return {
          tool,
          integrationType: assuredIntegrationType,
          isConnected: hasAllScopes,
          connectionId: hasAllScopes ? providerConn.connectionId : null,
          requiredScopes,
        };
      }
    }

    return {
      tool,
      integrationType: assuredIntegrationType,
      isConnected: false,
      connectionId: null,
      requiredScopes,
    };
  });
};

const buildConnectedIntegrations = (
  toolsWithStatus: ToolIntegrationStatus[],
  integrationProviderMap: Map<IntegrationType, string>,
): Map<IntegrationType, ConnectedAccount> => {
  const map = new Map<IntegrationType, ConnectedAccount>();
  toolsWithStatus.forEach((status) => {
    if (status.integrationType && status.isConnected && !map.has(status.integrationType)) {
      map.set(status.integrationType, {
        id: status.connectionId ?? '',
        status: 'ACTIVE',
        toolkit: {
          slug: integrationProviderMap.get(status.integrationType) || status.integrationType,
        },
      });
    }
  });
  return map;
};

const deriveToolGroups = ({
  tools,
  toolsWithStatus,
}: {
  tools: ToolMetadata[];
  toolsWithStatus: ToolIntegrationStatus[];
}): {
  toolsByIntegration: Map<string, ToolMetadata[]>;
  integrationGroups: IntegrationGroup[];
} => {
  const toolsByIntegration = new Map<string, ToolMetadata[]>();

  tools.forEach((tool) => {
    const integrationType = getIntegrationType(tool) || 'other';
    if (!toolsByIntegration.has(integrationType)) {
      toolsByIntegration.set(integrationType, []);
    }
    toolsByIntegration.get(integrationType)!.push(tool);
  });

  const integrationGroups: IntegrationGroup[] = [];
  toolsByIntegration.forEach((groupTools, integrationType) => {
    const firstTool = groupTools[0];
    const isConnected = toolsWithStatus.some(
      (status) =>
        (getIntegrationType(status.tool) || 'other') === integrationType && status.isConnected,
    );

    integrationGroups.push({
      integration_type: integrationType,
      tools: groupTools,
      display_name: getIntegrationDisplayName(integrationType),
      description: firstTool?.description || `${getIntegrationDisplayName(integrationType)} tools`,
      isConnected,
    });
  });

  integrationGroups.sort((a, b) => {
    if (a.isConnected !== b.isConnected) {
      return a.isConnected ? -1 : 1;
    }
    return a.display_name.localeCompare(b.display_name);
  });

  return {
    toolsByIntegration,
    integrationGroups,
  };
};

export const deriveIntegrationData = ({
  tools,
  toolStatus,
  connections,
}: {
  tools: ToolMetadata[];
  toolStatus: ToolConnectionStatus[] | null;
  connections: ConnectedAccount[];
}) => {
  const integrationProviderMap = buildIntegrationProviderMap(tools);
  const integrationScopesMap = buildIntegrationScopesMap(tools);
  const providerConnectionsMap = buildProviderConnectionsMap(connections);
  const toolStatusMap = buildToolStatusMap(toolStatus);
  const toolsWithStatus = computeToolsWithStatus({
    tools,
    toolStatusMap,
    providerConnectionsMap,
    integrationProviderMap,
    integrationScopesMap,
  });
  const connectedIntegrations = buildConnectedIntegrations(toolsWithStatus, integrationProviderMap);
  const { toolsByIntegration, integrationGroups } = deriveToolGroups({ tools, toolsWithStatus });

  return {
    integrationProviderMap,
    integrationScopesMap,
    providerConnectionsMap,
    toolStatusMap,
    toolsWithStatus,
    connectedIntegrations,
    toolsByIntegration,
    integrationGroups,
  };
};

export interface ProviderConnectionInfo {
  scopes: Set<string>;
  connectionId: string;
}
