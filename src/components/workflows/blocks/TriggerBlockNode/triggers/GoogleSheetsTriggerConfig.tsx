import { Sheet } from 'lucide-react';

import { useTriggerIntegration } from '@/hooks/useTriggerIntegration';
import { TriggerAccountSelector } from '../../../block-config/widgets/TriggerAccountSelector';

export const GoogleSheetsDetailsSection: React.FC<{ triggerKey: string; triggerId: string }> = ({ triggerKey, triggerId }) => {
  const sheetsIntegration = useTriggerIntegration(triggerKey, triggerId);

  return (
    <div className="rounded-md border border-dashed p-3 space-y-3 bg-muted/40">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Sheet className="h-4 w-4" />
        Google Sheets connection
      </div>
      <TriggerAccountSelector
        triggerKey={sheetsIntegration?.triggerKey ?? triggerKey}
        selectedConnectionId={sheetsIntegration?.connectionId}
        onSelect={sheetsIntegration?.onSelectAccount}
        onConnectAccount={sheetsIntegration?.onConnect}
      />
    </div>
  );
};
