import { CalendarDays } from 'lucide-react';

import { useTriggerIntegration } from '@/hooks/useTriggerIntegration';
import { TriggerAccountSelector } from '../../../block-config/widgets/TriggerAccountSelector';

export const GoogleCalendarDetailsSection: React.FC<{ triggerKey: string; triggerId: string }> = ({ triggerKey, triggerId }) => {
  const calendarIntegration = useTriggerIntegration(triggerKey, triggerId);

  return (
    <div className="rounded-md border border-dashed p-3 space-y-3 bg-muted/40">
      <div className="flex items-center gap-2 text-sm font-medium">
        <CalendarDays className="h-4 w-4" />
        Google Calendar connection
      </div>
      <TriggerAccountSelector
        triggerKey={calendarIntegration?.triggerKey ?? triggerKey}
        selectedConnectionId={calendarIntegration?.connectionId}
        onSelect={calendarIntegration?.onSelectAccount}
        onConnectAccount={calendarIntegration?.onConnect}
      />
    </div>
  );
};
