import { GMAIL_TOOL_FALLBACK_NAMES } from '@/components/workflows/triggers/constants';
import { useProviderIntegrationData } from './useProviderIntegrationData';

/**
 * Hook for Gmail integration OAuth flow.
 *
 * Note: Connection status (is_connected) is now provided by the backend
 * via trigger descriptors. This hook only provides data needed for the
 * OAuth connect flow (connectionId, toolNames).
 */
export function useGmailIntegration() {
  const { toolNames: gmailToolNames, connectionId: gmailConnectionId } = useProviderIntegrationData({
    integrationType: 'gmail',
    fallbackToolNames: GMAIL_TOOL_FALLBACK_NAMES,
  });

  return {
    gmailToolNames,
    gmailConnectionId,
  };
}
