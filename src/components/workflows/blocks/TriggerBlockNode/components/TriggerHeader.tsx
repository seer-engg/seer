import { Calendar, CalendarDays, Database, Link, Mail, MessageSquare, Hash, Table2, Sheet } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { useTriggerToggle } from '@/hooks/useTriggerToggle';
import type { WorkflowNodeData } from '../../../types';
import type { TriggerKind } from './constants';

const TRIGGER_ICONS: Record<TriggerKind, typeof Calendar> = {
  cron: Calendar,
  gmail: Mail,
  discord: MessageSquare,
  slack: Hash,
  supabase: Database,
  webhook: Link,
  form: Link,
  calendar: CalendarDays,
  airtable: Table2,
  google_sheets: Sheet,
};

interface TriggerHeaderProps {
  label: string;
  descriptor?: WorkflowNodeData['triggerMeta']['descriptor'] | null;
  subscription?: WorkflowNodeData['triggerMeta']['subscription'];
  triggerKey: string;
  triggerKind: TriggerKind;
}

export const TriggerHeader: React.FC<TriggerHeaderProps> = ({
  label,
  subscription,
  triggerKind,
}) => {
  const uiMeta = subscription?.ui_meta as Record<string, unknown> | undefined;
  const subscriptionId = (uiMeta?.subscription_id as number) ?? null;

  const { isEnabled, isPending, handleToggle } = useTriggerToggle(subscriptionId);

  const TriggerIcon = TRIGGER_ICONS[triggerKind] ?? Link;

  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-2 min-w-0">
        <TriggerIcon className="h-4 w-4 text-primary flex-shrink-0" />
        <p className="font-medium text-sm truncate">{label}</p>
      </div>

      {/* Enable/Disable toggle - only show if subscription is saved */}
      {subscriptionId && (
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex-shrink-0">
              <Switch
                checked={isEnabled}
                onCheckedChange={handleToggle}
                disabled={isPending}
                className="scale-75 origin-right"
                aria-label={isEnabled ? 'Disable trigger' : 'Enable trigger'}
              />
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>{isEnabled ? 'Disable trigger' : 'Enable trigger'}</p>
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
};
