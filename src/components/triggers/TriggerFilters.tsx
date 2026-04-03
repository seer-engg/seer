import { Search, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import type { TriggerSubscriptionFilters } from '@/types/triggers';
import { TRIGGER_ICON_BY_KEY } from '@/components/workflows/triggers/constants';

interface TriggerFiltersProps {
  filters: TriggerSubscriptionFilters;
  onFilterChange: <K extends keyof TriggerSubscriptionFilters>(
    key: K,
    value: TriggerSubscriptionFilters[K]
  ) => void;
  onClearFilters: () => void;
  triggerKeys: string[];
  workflows: { id: string; title: string }[];
}

/** Human-readable labels for trigger keys */
const TRIGGER_KEY_LABELS: Record<string, string> = {
  'webhook.generic': 'Webhook',
  'poll.gmail.email_received': 'Gmail',
  'schedule.cron': 'Schedule',
  'webhook.supabase.db_changes': 'Supabase',
  'form.hosted': 'Form',
  'poll.discord.message_received': 'Discord',
  'poll.slack.message_received': 'Slack',
};

export function TriggerFilters({
  filters,
  onFilterChange,
  onClearFilters,
  triggerKeys,
  workflows,
}: TriggerFiltersProps) {
  const hasFilters = filters.search || filters.trigger_key || filters.workflow_id;

  return (
    <div className="flex flex-wrap items-center gap-3">
      {/* Search input */}
      <div className="relative flex-1 min-w-[200px] max-w-sm">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search triggers..."
          value={filters.search ?? ''}
          onChange={(e) => onFilterChange('search', e.target.value || undefined)}
          className="pl-9"
        />
      </div>

      {/* Trigger type filter */}
      <Select
        value={filters.trigger_key ?? 'all'}
        onValueChange={(value) =>
          onFilterChange('trigger_key', value === 'all' ? undefined : value)
        }
      >
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="All types" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All types</SelectItem>
          {triggerKeys.map((key) => {
            const Icon = TRIGGER_ICON_BY_KEY[key];
            const label = TRIGGER_KEY_LABELS[key] ?? key;
            return (
              <SelectItem key={key} value={key}>
                <div className="flex items-center gap-2">
                  {Icon && <Icon className="h-4 w-4 text-muted-foreground" />}
                  <span>{label}</span>
                </div>
              </SelectItem>
            );
          })}
        </SelectContent>
      </Select>

      {/* Workflow filter */}
      <Select
        value={filters.workflow_id ?? 'all'}
        onValueChange={(value) =>
          onFilterChange('workflow_id', value === 'all' ? undefined : value)
        }
      >
        <SelectTrigger className="w-[200px]">
          <SelectValue placeholder="All workflows" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All workflows</SelectItem>
          {workflows.map((wf) => (
            <SelectItem key={wf.id} value={wf.id}>
              {wf.title}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Clear filters button */}
      {hasFilters && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onClearFilters}
          className="h-9 px-2 text-muted-foreground hover:text-foreground"
        >
          <X className="h-4 w-4 mr-1" />
          Clear
        </Button>
      )}
    </div>
  );
}
