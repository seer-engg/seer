import { useMemo } from 'react';

import type { IntegrationType } from '@/lib/integrations/client';

import { useToolingData } from './useToolingData';

const parseProviderConnectionId = (raw?: string | null): number | null => {
  if (!raw) {
    return null;
  }

  const segments = raw.split(':');
  const numeric = Number(segments[segments.length - 1]);
  return Number.isNaN(numeric) ? null : numeric;
};

interface UseProviderIntegrationDataOptions {
  integrationType: IntegrationType;
  fallbackToolNames: string[];
  toolNameMatch?: string;
}

export function useProviderIntegrationData({
  integrationType,
  fallbackToolNames,
  toolNameMatch,
}: UseProviderIntegrationDataOptions) {
  const { tools: integrationTools, getConnectionId } = useToolingData();

  const toolNames = useMemo(() => {
    const normalized = Array.isArray(integrationTools) ? integrationTools : [];
    const match = (toolNameMatch ?? integrationType).toLowerCase();
    const names = normalized
      .filter((tool) => {
        const integration = tool.integration_type?.toLowerCase();
        if (integration) {
          return integration === integrationType;
        }

        return tool.name.toLowerCase().includes(match);
      })
      .map((tool) => tool.name);

    return names.length > 0 ? names : fallbackToolNames;
  }, [fallbackToolNames, integrationTools, integrationType, toolNameMatch]);

  const connectionIdRaw = getConnectionId(integrationType);
  const connectionId = useMemo(
    () => parseProviderConnectionId(connectionIdRaw),
    [connectionIdRaw],
  );

  return {
    toolNames,
    connectionId,
  };
}
