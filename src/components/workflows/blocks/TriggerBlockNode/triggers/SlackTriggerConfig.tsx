import { useQuery } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Loader2, Hash, CheckCircle2, AlertTriangle } from 'lucide-react';
import { backendApiClient } from '@/lib/api-client';
import { resourceKeys } from '@/lib/query-keys';
import { useJoinSlackChannel } from '@/hooks/useSlackIntegration';
import { useTriggerIntegration } from '@/hooks/useTriggerIntegration';

import { TriggerAccountSelector } from '../../../block-config/widgets/TriggerAccountSelector';

interface ChannelResource {
  id: string;
  name: string;
  display_name: string;
  metadata?: {
    channel_id?: string;
    channel_name?: string;
    is_private?: boolean;
    is_member?: boolean;
    workspace_id?: string;
  };
}

export const SlackDetailsSection: React.FC<{ triggerKey: string; triggerId: string }> = ({ triggerKey, triggerId }) => {
  const slackIntegration = useTriggerIntegration(triggerKey, triggerId);

  return (
    <div className="rounded-md border border-dashed p-3 space-y-3 bg-muted/40">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Hash className="h-4 w-4" />
        Slack connection
      </div>
      <TriggerAccountSelector
        triggerKey={slackIntegration?.triggerKey ?? triggerKey}
        selectedConnectionId={slackIntegration?.connectionId}
        onSelect={slackIntegration?.onSelectAccount}
        onConnectAccount={slackIntegration?.onConnect}
      />
    </div>
  );
};

// Helper: Render loading state
const renderLoadingState = () => (
  <div className="flex items-center gap-2 text-xs text-muted-foreground">
    <Loader2 className="h-3 w-3 animate-spin" />
    Checking bot membership...
  </div>
);

// Helper: Render member status (bot is in channel)
const renderMemberStatus = (channelName: string) => (
  <div className="flex items-center gap-2 text-xs text-emerald-600 dark:text-emerald-400">
    <CheckCircle2 className="h-3.5 w-3.5" />
    Bot is in #{channelName}
  </div>
);

// Helper: Render not-in-channel warning with join action
interface NotInChannelProps {
  channelName: string;
  isPrivate: boolean;
  onJoin: () => void;
  isPending: boolean;
  error?: Error | null;
}

const renderNotInChannelWarning = ({ channelName, isPrivate, onJoin, isPending, error }: NotInChannelProps) => (
  <div className="rounded-md border border-amber-500/20 bg-amber-500/10 p-3 space-y-2">
    <div className="flex items-center gap-2 text-sm text-amber-600 dark:text-amber-400">
      <AlertTriangle className="h-4 w-4" />
      Bot is not in #{channelName}
    </div>
    {isPrivate ? (
      <p className="text-xs text-muted-foreground">
        Private channel - invite the bot manually from Slack.
      </p>
    ) : (
      <Button size="sm" onClick={onJoin} disabled={isPending}>
        {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        Add bot to channel
      </Button>
    )}
    {error && (
      <p className="text-xs text-destructive">
        Failed to join channel. {error.message}
      </p>
    )}
  </div>
);

/**
 * Shows bot membership status for a selected Slack channel.
 * Allows users to add the bot to the channel if not already a member.
 */
export const SlackChannelStatus: React.FC<{
  workspaceId?: string;
  channelId?: string;
}> = ({ workspaceId, channelId }) => {
  const joinChannel = useJoinSlackChannel();

  // Fetch channel list to get membership status
  const { data: channelData, isLoading } = useQuery({
    queryKey: resourceKeys.slackChannelsByWorkspace(workspaceId),
    queryFn: async () => {
      if (!workspaceId) return null;
      const dependsOn = JSON.stringify({ workspace_id: workspaceId });
      return backendApiClient.request<{ items: ChannelResource[] }>(
        `/api/integrations/resources/slack/channel?depends_on=${encodeURIComponent(dependsOn)}`,
      );
    },
    enabled: Boolean(workspaceId && channelId),
    staleTime: 30_000,
  });

  // Find the selected channel in the list
  const selectedChannel = channelData?.items?.find((ch) => ch.id === channelId);

  // Early returns for empty states
  if (!channelId || !workspaceId) return null;
  if (isLoading) return renderLoadingState();
  if (!selectedChannel) return null;

  const isMember = selectedChannel.metadata?.is_member;
  const isPrivate = selectedChannel.metadata?.is_private ?? false;
  const channelName = selectedChannel.metadata?.channel_name || selectedChannel.name;

  if (isMember) return renderMemberStatus(channelName);

  const handleJoinChannel = () => {
    joinChannel.mutate({ workspaceId, channelId });
  };

  return renderNotInChannelWarning({
    channelName,
    isPrivate,
    onJoin: handleJoinChannel,
    isPending: joinChannel.isPending,
    error: joinChannel.error as Error | null,
  });
};
