/**
 * RunOptionsDialog Component
 *
 * Dialog that appears when running a workflow with triggers, allowing users to
 * select a real event from connected accounts or stored events to run with.
 */

import { useMemo } from 'react';
import { Clock, Database, Mail, MessageSquare, Play, Zap, FileText, UserCheck, Loader2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { TriggerSpec } from '@/types/workflow-spec';
import type { TriggerBrowsingMode } from './TriggerEventPicker/types';

export interface TriggerWithConnection extends TriggerSpec {
  provider: string;
  /** OAuth connection ID for polling triggers */
  connectionId: number | null;
  /** Subscription ID for persisted triggers (webhooks, forms) */
  subscriptionId: number | null;
  /** Whether the trigger is connected (OAuth) or has events (persisted) */
  isConnected: boolean;
  /** Whether the trigger has browsable events */
  hasEvents: boolean;
  /** Browsing mode: 'polling' for OAuth, 'persisted' for webhooks/forms, null for cron */
  browsingMode: TriggerBrowsingMode | null;
}

export interface RunOptionsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  triggers: TriggerWithConnection[];
  onSelectTrigger: (trigger: TriggerWithConnection) => void;
  onConnectProvider?: (provider: string) => void;
  /** Handler for instant triggers (e.g., cron) - triggers workflow immediately */
  onTriggerNow?: (trigger: TriggerWithConnection) => void;
  /** Whether an instant trigger is currently being executed */
  isTriggering?: boolean;
}

function getTriggerIcon(triggerKey: string) {
  if (triggerKey.includes('gmail') || triggerKey.includes('email')) {
    return <Mail className="w-4 h-4 text-red-500" />;
  }
  if (triggerKey.includes('discord')) {
    return <MessageSquare className="w-4 h-4 text-indigo-500" />;
  }
  if (triggerKey.includes('supabase')) {
    return <Database className="w-4 h-4 text-emerald-500" />;
  }
  if (triggerKey.startsWith('form.hitl')) {
    return <UserCheck className="w-4 h-4 text-blue-500" />;
  }
  if (triggerKey.startsWith('form.')) {
    return <FileText className="w-4 h-4 text-purple-500" />;
  }
  if (triggerKey.startsWith('schedule.')) {
    return <Clock className="w-4 h-4 text-blue-500" />;
  }
  return <Zap className="w-4 h-4 text-amber-500" />;
}

function getTriggerLabel(triggerKey: string): string {
  if (triggerKey.includes('gmail')) return 'Gmail';
  if (triggerKey.includes('discord')) return 'Discord';
  if (triggerKey === 'webhook.supabase.db_changes') return 'Supabase';
  if (triggerKey.includes('webhook')) return 'Webhook';
  if (triggerKey === 'form.hitl') return 'HITL';
  if (triggerKey.startsWith('form.')) return 'Form';
  if (triggerKey === 'schedule.cron') return 'Scheduler';
  const parts = triggerKey.split('.');
  return parts[parts.length - 1]?.replace(/_/g, ' ') || 'Trigger';
}

function ConnectedTriggerItem({
  trigger,
  onSelect,
}: {
  trigger: TriggerWithConnection;
  onSelect: (t: TriggerWithConnection) => void;
}) {
  const isPersisted = trigger.browsingMode === 'persisted';
  const statusLabel = isPersisted ? 'Has Events' : 'Connected';
  const description = isPersisted
    ? 'Browse recently received events'
    : 'Browse recent events from your account';

  return (
    <Button
      variant="outline"
      className="w-full justify-start h-auto py-3 px-4 group"
      onClick={() => onSelect(trigger)}
    >
      {getTriggerIcon(trigger.key)}
      <div className="flex-1 text-left ml-3">
        <div className="font-medium flex items-center gap-2">
          {getTriggerLabel(trigger.key)}
          <Badge
            variant="secondary"
            className="h-4 px-1 text-[9px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
          >
            {statusLabel}
          </Badge>
        </div>
        <div className="text-xs text-muted-foreground">{description}</div>
      </div>
    </Button>
  );
}

function DisconnectedTriggerItem({
  trigger,
  onConnect,
}: {
  trigger: TriggerWithConnection;
  onConnect?: (provider: string) => void;
}) {
  const isPersisted = trigger.browsingMode === 'persisted';
  const statusLabel = isPersisted ? 'No Events' : 'Not Connected';
  const description = isPersisted
    ? 'Send a test event to browse real events'
    : 'Connect to browse real events';

  return (
    <div className={cn('flex items-center gap-3 p-3 rounded-md border border-dashed', 'bg-muted/30')}>
      {getTriggerIcon(trigger.key)}
      <div className="flex-1">
        <div className="font-medium text-sm flex items-center gap-2">
          {getTriggerLabel(trigger.key)}
          <Badge
            variant="secondary"
            className="h-4 px-1 text-[9px] bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
          >
            {statusLabel}
          </Badge>
        </div>
        <div className="text-xs text-muted-foreground">{description}</div>
      </div>
      {/* Only show Connect button for polling triggers that need OAuth */}
      {!isPersisted && onConnect && (
        <Button variant="default" size="sm" onClick={() => onConnect(trigger.provider)}>
          Connect
        </Button>
      )}
    </div>
  );
}

function InstantTriggerItem({
  trigger,
  onTriggerNow,
  isTriggering,
}: {
  trigger: TriggerWithConnection;
  onTriggerNow: (t: TriggerWithConnection) => void;
  isTriggering?: boolean;
}) {
  return (
    <Button
      variant="outline"
      className="w-full justify-start h-auto py-3 px-4 group"
      onClick={() => onTriggerNow(trigger)}
      disabled={isTriggering}
    >
      {isTriggering ? (
        <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
      ) : (
        getTriggerIcon(trigger.key)
      )}
      <div className="flex-1 text-left ml-3">
        <div className="font-medium flex items-center gap-2">
          {getTriggerLabel(trigger.key)}
          <Badge
            variant="secondary"
            className="h-4 px-1 text-[9px] bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20"
          >
            Ready
          </Badge>
        </div>
        <div className="text-xs text-muted-foreground">Trigger workflow immediately</div>
      </div>
      <Play className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
    </Button>
  );
}


export function RunOptionsDialog({
  open,
  onOpenChange,
  triggers,
  onSelectTrigger,
  onConnectProvider,
  onTriggerNow,
  isTriggering,
}: RunOptionsDialogProps) {
  // Instant triggers (e.g., cron) - can trigger immediately without browsing events
  const instantTriggers = useMemo(
    () => triggers.filter((t) => t.browsingMode === 'instant'),
    [triggers],
  );

  // Triggers that can browse events:
  // - Polling triggers with connection (isConnected)
  // - Persisted triggers with events (hasEvents)
  const browsableTriggers = useMemo(
    () =>
      triggers.filter((t) => {
        // Skip cron/instant triggers (no event browsing)
        if (t.browsingMode === null || t.browsingMode === 'instant') return false;
        // For polling, need to be connected
        if (t.browsingMode === 'polling') return t.isConnected;
        // For persisted, need to have events
        if (t.browsingMode === 'persisted') return t.hasEvents;
        return false;
      }),
    [triggers],
  );

  // Triggers that can't browse events yet:
  // - Polling triggers without connection
  // - Persisted triggers without events
  const nonBrowsableTriggers = useMemo(
    () =>
      triggers.filter((t) => {
        // Skip cron/instant triggers
        if (t.browsingMode === null || t.browsingMode === 'instant') return false;
        if (t.browsingMode === 'polling') return !t.isConnected;
        if (t.browsingMode === 'persisted') return !t.hasEvents;
        return false;
      }),
    [triggers],
  );

  const hasAnyTriggers = instantTriggers.length > 0 || browsableTriggers.length > 0 || nonBrowsableTriggers.length > 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Select Trigger Event</DialogTitle>
          <DialogDescription>
            Select a real event from your connected accounts to test this workflow.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* Instant triggers (cron) - shown first as they're ready to run */}
          {instantTriggers.length > 0 && onTriggerNow && (
            <div className="space-y-2">
              {instantTriggers.map((trigger) => (
                <InstantTriggerItem
                  key={trigger.id}
                  trigger={trigger}
                  onTriggerNow={onTriggerNow}
                  isTriggering={isTriggering}
                />
              ))}
            </div>
          )}

          {browsableTriggers.length > 0 && (
            <div className="space-y-2">
              {browsableTriggers.map((trigger) => (
                <ConnectedTriggerItem key={trigger.id} trigger={trigger} onSelect={onSelectTrigger} />
              ))}
            </div>
          )}

          {nonBrowsableTriggers.length > 0 && (
            <div className="space-y-2">
              {nonBrowsableTriggers.map((trigger) => (
                <DisconnectedTriggerItem key={trigger.id} trigger={trigger} onConnect={onConnectProvider} />
              ))}
            </div>
          )}

          {!hasAnyTriggers && (
            <div className="text-center py-4 text-muted-foreground">
              No browsable triggers found. Connect a trigger or send test events to get started.
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
