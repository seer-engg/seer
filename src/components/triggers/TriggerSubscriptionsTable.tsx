import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import { Zap, ExternalLink, Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useWorkflowStore } from '@/stores/workflowStore';
import { generateWorkflowName } from '@/lib/workflow-names';
import { toast } from '@/components/ui/sonner';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Switch } from '@/components/ui/switch';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import type { TriggerSubscriptionListItem } from '@/types/triggers';
import {
  TRIGGER_ICON_BY_KEY,
  TRIGGER_COLOR_BY_KEY,
} from '@/components/workflows/triggers/constants';

interface TriggerSubscriptionsTableProps {
  subscriptions: TriggerSubscriptionListItem[];
  isLoading: boolean;
  onToggleEnabled: (subscriptionId: number, enabled: boolean) => void;
  isToggling?: boolean;
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

function LoadingSkeleton() {
  return (
    <>
      {[1, 2, 3, 4, 5].map((i) => (
        <TableRow key={i}>
          <TableCell>
            <div className="flex items-center gap-3">
              <Skeleton className="h-8 w-8 rounded-md" />
              <div className="space-y-1">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-3 w-16" />
              </div>
            </div>
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-32" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-4 w-24" />
          </TableCell>
          <TableCell>
            <Skeleton className="h-6 w-11 rounded-full" />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
}

function EmptyState() {
  const navigate = useNavigate();
  const [isCreating, setIsCreating] = useState(false);
  const createWorkflow = useWorkflowStore((state) => state.createWorkflow);

  const handleCreateWorkflow = async () => {
    if (isCreating) return;
    setIsCreating(true);
    try {
      const workflow = await createWorkflow(generateWorkflowName(), { nodes: [], edges: [] });
      navigate(`/workflows/${workflow.workflow_id}`, { replace: true });
      toast.success('Workflow created');
    } catch (error) {
      console.error('Failed to create workflow:', error);
      toast.error('Failed to create workflow');
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="rounded-full bg-muted p-4 mb-4">
        <Zap className="h-8 w-8 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-medium mb-1">No triggers found</h3>
      <p className="text-sm text-muted-foreground max-w-sm mb-6">
        Triggers are created within workflow editors. Add a trigger to a workflow to see it here.
      </p>
      <Button onClick={handleCreateWorkflow} disabled={isCreating}>
        <Plus className="h-4 w-4 mr-2" />
        {isCreating ? 'Creating...' : 'Create Workflow'}
      </Button>
    </div>
  );
}

function TriggerTypeCell({ triggerKey }: { triggerKey: string }) {
  const Icon = TRIGGER_ICON_BY_KEY[triggerKey] ?? Zap;
  const colors = TRIGGER_COLOR_BY_KEY[triggerKey] ?? {
    bg: 'bg-muted',
    text: 'text-muted-foreground',
  };
  const label = TRIGGER_KEY_LABELS[triggerKey] ?? triggerKey;

  return (
    <div className="flex items-center gap-3">
      <div className={cn('p-2 rounded-md', colors.bg)}>
        <Icon className={cn('h-4 w-4', colors.text)} />
      </div>
      <span className="text-sm font-medium">{label}</span>
    </div>
  );
}

function LastEventCell({ lastEventAt }: { lastEventAt: string | null }) {
  if (!lastEventAt) {
    return <span className="text-muted-foreground text-sm">Never</span>;
  }

  const date = new Date(lastEventAt);
  const relative = formatDistanceToNow(date, { addSuffix: true });

  return (
    <span className="text-sm" title={date.toLocaleString()}>
      {relative}
    </span>
  );
}

export function TriggerSubscriptionsTable({
  subscriptions,
  isLoading,
  onToggleEnabled,
  isToggling,
}: TriggerSubscriptionsTableProps) {
  if (!isLoading && subscriptions.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[200px]">Type</TableHead>
            <TableHead>Workflow</TableHead>
            <TableHead className="w-[150px]">Last Event</TableHead>
            <TableHead className="w-[100px] text-right">Enabled</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {isLoading ? (
            <LoadingSkeleton />
          ) : (
            subscriptions.map((sub) => (
              <TableRow key={sub.id}>
                <TableCell>
                  <TriggerTypeCell triggerKey={sub.trigger_key} />
                </TableCell>
                <TableCell>
                  <Link
                    to={`/workflows/${sub.workflow_id}`}
                    className="group flex items-center gap-1 text-sm font-medium text-primary hover:underline"
                  >
                    {sub.workflow_title}
                    <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </Link>
                  {sub.title && (
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {sub.title}
                    </p>
                  )}
                </TableCell>
                <TableCell>
                  <LastEventCell lastEventAt={sub.last_event_at} />
                </TableCell>
                <TableCell className="text-right">
                  <Switch
                    checked={sub.enabled}
                    onCheckedChange={(checked) => onToggleEnabled(sub.id, checked)}
                    disabled={isToggling}
                    aria-label={sub.enabled ? 'Disable trigger' : 'Enable trigger'}
                  />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
