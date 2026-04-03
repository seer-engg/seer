import { MessageSquare } from 'lucide-react';

import { useTriggerIntegration } from '@/hooks/useTriggerIntegration';
import { TriggerAccountSelector } from '../../../block-config/widgets/TriggerAccountSelector';

export const DiscordDetailsSection: React.FC<{ triggerKey: string; triggerId: string }> = ({ triggerKey, triggerId }) => {
  const discordIntegration = useTriggerIntegration(triggerKey, triggerId);

  return (
    <div className="rounded-md border border-dashed p-3 space-y-3 bg-muted/40">
      <div className="flex items-center gap-2 text-sm font-medium">
        <MessageSquare className="h-4 w-4" />
        Discord connection
      </div>
      <TriggerAccountSelector
        triggerKey={discordIntegration?.triggerKey ?? triggerKey}
        selectedConnectionId={discordIntegration?.connectionId}
        onSelect={discordIntegration?.onSelectAccount}
        onConnectAccount={discordIntegration?.onConnect}
      />
    </div>
  );
};
