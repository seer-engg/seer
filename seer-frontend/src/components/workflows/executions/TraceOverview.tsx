import { useState, useRef, useEffect } from 'react';
import { format } from 'date-fns';
import { Copy, Check, AlertCircle, Square } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import type { RunHistoryEntry } from './types';

interface TraceOverviewProps {
  entry: RunHistoryEntry;
  onCancel?: () => void;
}

function calculateDuration(startedAt?: string | null, finishedAt?: string | null): string {
  if (!startedAt || !finishedAt) return '-';
  const seconds = Math.round(
    (new Date(finishedAt).getTime() - new Date(startedAt).getTime()) / 1000
  );
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

function getStatusBadgeClasses(status: string) {
  switch (status) {
    case 'succeeded':
      return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20';
    case 'failed':
      return 'bg-bug/10 text-bug dark:text-bug border-bug/20';
    case 'running':
      return 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20';
    case 'queued':
      return 'bg-muted/10 text-muted-foreground border-muted/20';
    case 'cancelled':
      return 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20';
    default:
      return 'bg-muted/10 text-muted-foreground border-muted/20';
  }
}

function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); }, []);

  const handleCopy = async () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    await navigator.clipboard.writeText(content);
    setCopied(true);
    timeoutRef.current = setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-6 w-6 shrink-0"
      onClick={handleCopy}
      aria-label={copied ? 'Copied' : 'Copy to clipboard'}
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
    </Button>
  );
}

export function MetadataTable({ entry, onCancel }: TraceOverviewProps) {
  const duration = calculateDuration(entry.started_at, entry.finished_at);
  const successCount = entry.nodes?.filter((n) => !n.error).length || 0;
  const failedCount = entry.nodes?.filter((n) => !!n.error).length || 0;

  return (
    <table className="w-full text-sm">
      <tbody>
        <tr className="border-b border-border/50">
          <td className="py-1.5 pr-4 text-muted-foreground w-32">Run ID</td>
          <td className="py-1.5 font-mono text-xs">{entry.run_id}</td>
        </tr>
        <tr className="border-b border-border/50">
          <td className="py-1.5 pr-4 text-muted-foreground">Status</td>
          <td className="py-1.5 flex items-center gap-2">
            <Badge variant="secondary" className={getStatusBadgeClasses(entry.status)}>
              {entry.status.charAt(0).toUpperCase() + entry.status.slice(1)}
            </Badge>
            {onCancel && ['running', 'queued', 'interrupted'].includes(entry.status) && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs gap-1 text-destructive hover:text-destructive hover:bg-destructive/10"
                onClick={onCancel}
              >
                <Square className="w-3 h-3" />
                Cancel
              </Button>
            )}
          </td>
        </tr>
        {entry.trigger && (
          <>
            <tr className="border-b border-border/50">
              <td className="py-1.5 pr-4 text-muted-foreground">Triggered By</td>
              <td className="py-1.5 flex items-center gap-2">
                <Badge variant="secondary" className="bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20">
                  {entry.trigger.title}
                </Badge>
                <span className="font-mono text-xs text-muted-foreground">{entry.trigger.trigger_key}</span>
              </td>
            </tr>
            {entry.trigger.occurred_at && (
              <tr className="border-b border-border/50">
                <td className="py-1.5 pr-4 text-muted-foreground">Triggered At</td>
                <td className="py-1.5">
                  {format(new Date(entry.trigger.occurred_at), 'MMM d, h:mm:ss a')}
                </td>
              </tr>
            )}
          </>
        )}
        <tr className="border-b border-border/50">
          <td className="py-1.5 pr-4 text-muted-foreground">Start Time</td>
          <td className="py-1.5">
            {entry.started_at
              ? format(new Date(entry.started_at), 'MMM d, h:mm:ss a')
              : '-'}
          </td>
        </tr>
        <tr className="border-b border-border/50">
          <td className="py-1.5 pr-4 text-muted-foreground">End Time</td>
          <td className="py-1.5">
            {entry.finished_at
              ? format(new Date(entry.finished_at), 'MMM d, h:mm:ss a')
              : '-'}
          </td>
        </tr>
        <tr className="border-b border-border/50">
          <td className="py-1.5 pr-4 text-muted-foreground">Duration</td>
          <td className="py-1.5">{duration}</td>
        </tr>
        {entry.nodes && entry.nodes.length > 0 && (
          <tr>
            <td className="py-1.5 pr-4 text-muted-foreground">Nodes</td>
            <td className="py-1.5">
              {entry.nodes.length} total
              {successCount > 0 && (
                <span className="text-emerald-600 dark:text-emerald-400 ml-2">
                  • {successCount} succeeded
                </span>
              )}
              {failedCount > 0 && (
                <span className="text-bug ml-2">• {failedCount} failed</span>
              )}
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

export function WorkflowIO({ entry }: TraceOverviewProps) {
  return (
    <Accordion type="multiple" className="w-full">
      {entry.inputs && Object.keys(entry.inputs).length > 0 && (
        <AccordionItem value="inputs">
          <AccordionTrigger className="text-sm">
            <div className="flex items-center justify-between w-full pr-2">
              <span>Workflow Inputs</span>
              <CopyButton content={JSON.stringify(entry.inputs, null, 2)} />
            </div>
          </AccordionTrigger>
          <AccordionContent className="items-start">
            <pre className="text-xs text-left bg-muted p-3 rounded-md overflow-x-auto scrollbar-thin">
              {JSON.stringify(entry.inputs, null, 2)}
            </pre>
          </AccordionContent>
        </AccordionItem>
      )}
      {entry.output && Object.keys(entry.output).length > 0 && (
        <AccordionItem value="output">
          <AccordionTrigger className="text-sm">
            <div className="flex items-center justify-between w-full pr-2">
              <span>Workflow Output</span>
              <CopyButton content={JSON.stringify(entry.output, null, 2)} />
            </div>
          </AccordionTrigger>
          <AccordionContent className="items-start">
            <pre className="text-xs text-left bg-muted p-3 rounded-md overflow-x-auto scrollbar-thin">
              {JSON.stringify(entry.output, null, 2)}
            </pre>
          </AccordionContent>
        </AccordionItem>
      )}
    </Accordion>
  );
}

export function TraceOverview({ entry, onCancel }: TraceOverviewProps) {
  return (
    <div className="space-y-4">
      <MetadataTable entry={entry} onCancel={onCancel} />
      {entry.error && (
        <div className="rounded-md border border-bug/20 bg-bug/5 p-3 space-y-1">
          <div className="flex items-center gap-2 text-sm font-medium text-bug">
            <AlertCircle className="h-4 w-4" />
            <span>Run Error</span>
          </div>
          <p className="text-xs text-muted-foreground font-mono break-all">{entry.error}</p>
        </div>
      )}
      <WorkflowIO entry={entry} />
    </div>
  );
}
