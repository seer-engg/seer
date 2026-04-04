import { useCallback } from 'react';
import { useCurrentUser } from '@/hooks/useAuthProvider';

import { initiateConnection } from '@/lib/api-client';
import { IntegrationType, formatScopes } from '@/lib/integrations/client';
import {
  getIntegrationProviderForType,
  getIntegrationScopesForType,
  integrationNeedsConnection,
} from '@/stores/toolsStore.helpers';

import { useToolingData } from './useToolingData';

export function useConnectIntegration() {
  const { user } = useCurrentUser();
  const userEmail = user?.primaryEmailAddress?.emailAddress ?? user?.emailAddresses?.[0]?.emailAddress ?? null;
  const { tools, integrationProviderMap, integrationScopesMap } = useToolingData();

  return useCallback(async (
    type: IntegrationType,
    options: { toolNames?: string[]; toolName?: string; forceNewAccount?: boolean; connectionId?: number },
  ): Promise<string | null> => {
    if (!userEmail) {
      console.error('[useConnectIntegration] Cannot connect integration without user email.');
      return null;
    }
    if (!integrationNeedsConnection(type)) {
      console.warn(`[useConnectIntegration] Integration ${type} does not require OAuth connection.`);
      return null;
    }

    const toolNames = options.toolNames ?? (options.toolName ? [options.toolName] : []);
    if (!toolNames.length) {
      console.error(`[useConnectIntegration] connectIntegration requires toolNames for ${type}.`);
      return null;
    }

    const scopeSet = new Set<string>();

    if (type === 'discord') {
      toolNames.forEach((toolName) => scopeSet.add(toolName));
    } else {
      toolNames.forEach((toolName) => {
        const tool = tools.find((candidate) => candidate.name === toolName);
        if (tool?.required_scopes?.length) {
          tool.required_scopes.forEach((scope) => scopeSet.add(scope));
        }
      });
    }

    if (!scopeSet.size) {
      getIntegrationScopesForType(type, integrationScopesMap).forEach((scope) => scopeSet.add(scope));
    }

    const scopelessProviders = new Set(['notion']);
    if (!scopeSet.size && !scopelessProviders.has(type)) {
      console.error(`[useConnectIntegration] No scopes available for integration type ${type}.`);
      return null;
    }

    const provider = getIntegrationProviderForType(type, integrationProviderMap);
    if (!provider && type !== 'sandbox') {
      console.error(`[useConnectIntegration] No OAuth provider configured for ${type}`);
      return null;
    }

    const callbackUrl = `${window.location.origin}${window.location.pathname}`;
    const response = await initiateConnection({
      userId: userEmail,
      provider: provider ?? type,
      scope: formatScopes(Array.from(scopeSet)),
      callbackUrl,
      integrationType: type,
      forceNewAccount: options.forceNewAccount,
      connectionId: options.connectionId,
    });

    return response.redirectUrl;
  }, [integrationProviderMap, integrationScopesMap, tools, userEmail]);
}
