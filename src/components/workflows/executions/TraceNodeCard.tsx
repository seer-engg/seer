import { useState, useRef, useEffect } from 'react';
import { format } from 'date-fns';
import { AlertCircle, CheckCircle2, Copy, Check, Code, List, Zap } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { JsonTreeView, JsonRawView } from './JsonTreeView';
import { getNodeDisplayName } from './timing-utils';
import { FixWithAI } from '@/components/ui/fix-with-ai';
import { ArtifactList } from './ArtifactList';
import type { WorkflowNodeTrace, RunHistoryEntry, TriggerInfo } from './types';

interface TraceNodeCardProps {
  node: WorkflowNodeTrace;
  index: number;
  executionGraph?: RunHistoryEntry['execution_graph'];
  workflowId?: string;
  triggerInfo?: TriggerInfo;
}

type ViewMode = 'formatted' | 'raw';

// Copy button
function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); }, []);

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    await navigator.clipboard.writeText(content);
    setCopied(true);
    timeoutRef.current = setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0" onClick={handleCopy}>
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
    </Button>
  );
}

// View mode toggle
function ViewModeToggle({ viewMode, onChange }: { viewMode: ViewMode; onChange: (mode: ViewMode) => void }) {
  const handleClick = (mode: ViewMode) => (e: React.MouseEvent) => {
    e.stopPropagation();
    onChange(mode);
  };

  return (
    <div className="flex items-center bg-muted rounded-md p-0.5" onClick={(e) => e.stopPropagation()}>
      <Button
        variant="ghost"
        size="sm"
        className={cn('h-6 px-2 text-xs gap-1', viewMode === 'formatted' && 'bg-background shadow-sm')}
        onClick={handleClick('formatted')}
      >
        <List className="h-3 w-3" />
        Formatted
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className={cn('h-6 px-2 text-xs gap-1', viewMode === 'raw' && 'bg-background shadow-sm')}
        onClick={handleClick('raw')}
      >
        <Code className="h-3 w-3" />
        Raw
      </Button>
    </div>
  );
}

// Error display
function NodeError({ error, workflowId }: { error: WorkflowNodeTrace['error']; workflowId?: string }) {
  if (!error) return null;
  const message = typeof error === 'string' ? error : error.message;
  return (
    <div className="bg-bug/10 border border-bug/20 rounded-md p-2 space-y-1">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-bug">{message}</p>
        <FixWithAI error={message} workflowId={workflowId} className="shrink-0" />
      </div>
    </div>
  );
}

// Data section with accordion
function DataAccordion({ title, data, viewMode }: { title: string; data: unknown; viewMode: ViewMode }) {
  if (!data || (typeof data === 'object' && Object.keys(data as object).length === 0)) {
    return null;
  }

  const jsonString = typeof data === 'string' ? data : JSON.stringify(data, null, 2);

  return (
    <AccordionItem value={title.toLowerCase()} className="border-none">
      <AccordionTrigger className="text-xs py-2 hover:no-underline">
        <div className="flex items-center justify-between w-full pr-2">
          <span>{title}</span>
          <CopyButton content={jsonString} />
        </div>
      </AccordionTrigger>
      <AccordionContent>
        <div className="max-h-60 overflow-y-auto bg-muted/30 rounded-md p-2">
          {viewMode === 'formatted' ? <JsonTreeView data={data} /> : <JsonRawView data={data} />}
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}

function TriggerCardView({ triggerInfo }: { triggerInfo: TriggerInfo }) {
  const [viewMode, setViewMode] = useState<ViewMode>('formatted');
  const occurredMs = triggerInfo.occurred_at ? new Date(triggerInfo.occurred_at).getTime() : null;
  const receivedMs = triggerInfo.received_at ? new Date(triggerInfo.received_at).getTime() : null;
  const latencyMs = occurredMs && receivedMs ? receivedMs - occurredMs : null;
  const hasEventData = !!triggerInfo.event_data && Object.keys(triggerInfo.event_data).length > 0;
  const eventDataJson = hasEventData ? JSON.stringify(triggerInfo.event_data, null, 2) : '';

  return (
    <Card className="p-3 border-l-4 border-l-blue-500/50">
      <div className="space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <Zap className="w-4 h-4 text-blue-500 shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium truncate">{triggerInfo.title}</p>
              <p className="text-xs text-muted-foreground font-mono">{triggerInfo.trigger_key}</p>
            </div>
          </div>
          <Badge variant="secondary" className="bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20 shrink-0">
            Trigger
          </Badge>
        </div>
        <div className="text-xs text-muted-foreground space-y-0.5">
          {triggerInfo.occurred_at && (
            <p>Occurred: <span className="text-foreground">{format(new Date(triggerInfo.occurred_at), 'h:mm:ss.SSS a')}</span></p>
          )}
          {triggerInfo.received_at && (
            <p>
              Received: <span className="text-foreground">{format(new Date(triggerInfo.received_at), 'h:mm:ss.SSS a')}</span>
              {latencyMs !== null && <span className="ml-1 text-blue-500">(+{latencyMs}ms)</span>}
            </p>
          )}
        </div>
        {hasEventData && (
          <Accordion type="multiple" defaultValue={['event-data']} className="w-full">
            <AccordionItem value="event-data" className="border-none">
              <AccordionTrigger className="text-xs py-2 hover:no-underline">
                <div className="flex items-center justify-between w-full pr-2">
                  <span>Event Data</span>
                  <div onClick={(e) => e.stopPropagation()} className="flex items-center gap-1">
                    <ViewModeToggle viewMode={viewMode} onChange={setViewMode} />
                    <CopyButton content={eventDataJson} />
                  </div>
                </div>
              </AccordionTrigger>
              <AccordionContent>
                <div className="max-h-60 overflow-y-auto bg-muted/30 rounded-md p-2">
                  {viewMode === 'formatted'
                    ? <JsonTreeView data={triggerInfo.event_data} />
                    : <JsonRawView data={triggerInfo.event_data} />}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        )}
      </div>
    </Card>
  );
}

export function TraceNodeCard({ node, index, executionGraph, workflowId, triggerInfo }: TraceNodeCardProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('formatted');
  const isSyntheticTrigger = node.is_synthetic && node.node_type === 'trigger';
  const hasError = !!node.error;
  const displayName = getNodeDisplayName(node.node_id, node.node_type, executionGraph);
  const hasData = (node.inputs && Object.keys(node.inputs).length > 0) || node.output;

  if (isSyntheticTrigger && triggerInfo) {
    return <TriggerCardView triggerInfo={triggerInfo} />;
  }

  return (
    <Card className={cn('p-3', hasError ? 'border-l-4 border-l-bug' : 'border-l-4 border-l-emerald-500/50')}>
      <div className="space-y-2">
        {/* Header */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            {hasError ? (
              <AlertCircle className="w-4 h-4 text-bug shrink-0" />
            ) : (
              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
            )}
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium truncate">Node {index + 1}: {displayName}</p>
              {node.timestamp && (
                <p className="text-xs text-muted-foreground">
                  {format(new Date(node.timestamp), 'h:mm:ss a')}
                  {node.tool_name && <span className="ml-2 font-mono">{node.tool_name}</span>}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {hasData && <ViewModeToggle viewMode={viewMode} onChange={setViewMode} />}
            <Badge
              variant="secondary"
              className={cn(
                hasError
                  ? 'bg-bug/10 text-bug border-bug/20'
                  : 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20'
              )}
            >
              {hasError ? 'Failed' : 'Success'}
            </Badge>
          </div>
        </div>

        {/* Error display */}
        {hasError && <NodeError error={node.error} workflowId={workflowId} />}

        {/* Data sections */}
        <Accordion type="multiple" defaultValue={['inputs', 'output']} className="w-full">
          <DataAccordion title="Inputs" data={node.inputs} viewMode={viewMode} />
          <DataAccordion title="Output" data={node.output} viewMode={viewMode} />
        </Accordion>
        <ArtifactList artifacts={node.artifacts} />
      </div>
    </Card>
  );
}
