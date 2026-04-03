import { DISCORD_TOOL_FALLBACK_NAMES } from '@/components/workflows/triggers/constants';
import { useProviderIntegrationData } from './useProviderIntegrationData';

/**
 * Hook for Discord integration OAuth flow.
 *
 * Note: Connection status (is_connected) is now provided by the backend
 * via trigger descriptors. This hook only provides data needed for the
 * OAuth connect flow (connectionId, toolNames).
 */
export function useDiscordIntegration() {
  const { toolNames: discordToolNames, connectionId: discordConnectionId } = useProviderIntegrationData({
    integrationType: 'discord',
    fallbackToolNames: DISCORD_TOOL_FALLBACK_NAMES,
  });

  return {
    discordToolNames,
    discordConnectionId,
  };
}
