import { useMutation, useQueryClient } from '@tanstack/react-query';
import { SLACK_TOOL_FALLBACK_NAMES } from '@/components/workflows/triggers/constants';
import { backendApiClient } from '@/lib/api-client';
import { resourceKeys } from '@/lib/query-keys';
import { useProviderIntegrationData } from './useProviderIntegrationData';

/**
 * Hook for Slack integration OAuth flow.
 *
 * Note: Connection status (is_connected) is now provided by the backend
 * via trigger descriptors. This hook only provides data needed for the
 * OAuth connect flow (connectionId, toolNames).
 */
export function useSlackIntegration() {
  const { toolNames: slackToolNames, connectionId: slackConnectionId } = useProviderIntegrationData({
    integrationType: 'slack',
    fallbackToolNames: SLACK_TOOL_FALLBACK_NAMES,
  });

  return {
    slackToolNames,
    slackConnectionId,
  };
}

/**
 * Hook for joining the Slack bot to a channel.
 *
 * Used in trigger configuration to allow users to add the bot
 * to a channel they've selected for monitoring.
 */
export function useJoinSlackChannel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      workspaceId,
      channelId,
    }: {
      workspaceId: string;
      channelId: string;
    }) => {
      return backendApiClient.request<{ ok: boolean; channel: Record<string, unknown> }>(
        `/api/integrations/resources/slack/channel/${channelId}/join?workspace_id=${encodeURIComponent(workspaceId)}`,
        { method: 'POST' },
      );
    },
    onSuccess: () => {
      // Invalidate channel resources to refresh is_member status
      queryClient.invalidateQueries({ queryKey: resourceKeys.slackChannels() });
    },
  });
}
