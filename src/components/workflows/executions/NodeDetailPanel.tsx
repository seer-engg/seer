import { useState, useRef, useEffect } from 'react';
import { format } from 'date-fns';
import { AlertCircle, CheckCircle2, Copy, Check, ChevronDown, Zap, Bot, Wrench, MessageSquare } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import { JsonTreeView } from './JsonTreeView';
import { getNodeDisplayName } from './timing-utils';
import { FixWithAI } from '@/components/ui/fix-with-ai';
import { ArtifactList } from './ArtifactList';
import type { WorkflowNodeTrace, RunHistoryEntry, TriggerInfo, AgentStep } from './types';

interface NodeDetailPanelProps {
  selectedNode: WorkflowNodeTrace;
  selectedIndex: number;
  executionGraph?: RunHistoryEntry['execution_graph'];
  workflowId?: string;
  triggerInfo?: TriggerInfo;
}

// Copy button
function CopyButton({ content, className }: { content: string; className?: string }) {
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
    <Button variant="ghost" size="icon" className={cn('h-6 w-6', className)} onClick={handleCopy}>
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
    </Button>
  );
}

// Node header with status
function NodeHeader({
  node,
  index,
  executionGraph,
  workflowId,
}: {
  node: WorkflowNodeTrace;
  index: number;
  executionGraph?: RunHistoryEntry['execution_graph'];
  workflowId?: string;
}) {
  const hasError = !!node.error;
  const displayName = getNodeDisplayName(node.node_id, node.node_type, executionGraph);

  return (
    <div className={cn('p-3 rounded-lg border', hasError ? 'border-bug/30 bg-bug/5' : 'border-border bg-muted/30')}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {hasError ? (
            <AlertCircle className="w-4 h-4 text-bug shrink-0" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
          )}
          <div className="min-w-0">
            <p className="text-sm font-medium truncate">
              Node {index + 1}: {displayName}
            </p>
            {node.timestamp && (
              <p className="text-xs text-muted-foreground">
                {format(new Date(node.timestamp), 'h:mm:ss a')}
                {node.tool_name && <span className="ml-2 font-mono">{node.tool_name}</span>}
                {node.model && <span className="ml-2 font-mono">{node.model}</span>}
              </p>
            )}
          </div>
        </div>
        <Badge
          variant="secondary"
          className={cn(
            'shrink-0',
            hasError
              ? 'bg-bug/10 text-bug border-bug/20'
              : 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20'
          )}
        >
          {hasError ? 'Failed' : 'Success'}
        </Badge>
      </div>

      {hasError && node.error && (
        <div className="mt-2 p-2 bg-bug/10 rounded text-xs text-bug space-y-1">
          <div className="flex items-center justify-between gap-2">
            <span>{typeof node.error === 'string' ? node.error : node.error.message}</span>
            <FixWithAI
              error={typeof node.error === 'string' ? node.error : node.error.message}
              workflowId={workflowId}
              className="shrink-0"
            />
          </div>
        </div>
      )}
    </div>
  );
}

// Collapsible data section
function DataSection({
  title,
  data,
  defaultOpen = true,
}: {
  title: string;
  data: unknown;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const jsonString = typeof data === 'string' ? data : JSON.stringify(data, null, 2);

  if (!data || (typeof data === 'object' && Object.keys(data as object).length === 0)) {
    return null;
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="rounded-lg border bg-card">
        <CollapsibleTrigger asChild>
          <button className="flex items-center justify-between w-full p-3 text-left hover:bg-muted/50 transition-colors">
            <div className="flex items-center gap-2">
              <ChevronDown className={cn('h-4 w-4 transition-transform', !isOpen && '-rotate-90')} />
              <span className="text-sm font-medium">{title}</span>
            </div>
            <CopyButton content={jsonString} className="opacity-60 hover:opacity-100" />
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="px-3 pb-3 pt-0">
            <div className="rounded-md bg-muted/30 p-3 max-h-[300px] overflow-y-auto text-left">
              <JsonTreeView data={data} />
            </div>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

// Trigger detail view for synthetic trigger nodes
function TriggerDetailView({ triggerInfo }: { triggerInfo: TriggerInfo }) {
  const occurredMs = triggerInfo.occurred_at ? new Date(triggerInfo.occurred_at).getTime() : null;
  const receivedMs = triggerInfo.received_at ? new Date(triggerInfo.received_at).getTime() : null;
  const latencyMs = occurredMs && receivedMs ? receivedMs - occurredMs : null;

  return (
    <div className="space-y-3">
      <div className="p-3 rounded-lg border border-blue-500/30 bg-blue-500/10">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Zap className="w-4 h-4 text-blue-500 shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-medium">{triggerInfo.title}</p>
              <p className="text-xs text-muted-foreground font-mono">{triggerInfo.trigger_id} · {triggerInfo.trigger_key}</p>
            </div>
          </div>
          <Badge variant="secondary" className="bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20 shrink-0">
            Trigger
          </Badge>
        </div>
        <div className="mt-2 space-y-0.5 text-xs text-muted-foreground">
          {triggerInfo.occurred_at && (
            <p>Occurred: <span className="text-foreground">{format(new Date(triggerInfo.occurred_at), 'MMM d, h:mm:ss.SSS a')}</span></p>
          )}
          {triggerInfo.received_at && (
            <p>
              Received: <span className="text-foreground">{format(new Date(triggerInfo.received_at), 'MMM d, h:mm:ss.SSS a')}</span>
              {latencyMs !== null && (
                <span className="ml-1 text-blue-500">(+{latencyMs}ms)</span>
              )}
            </p>
          )}
        </div>
      </div>
      {triggerInfo.event_data && Object.keys(triggerInfo.event_data).length > 0 && (
        <DataSection title="Event Data" data={triggerInfo.event_data} />
      )}
    </div>
  );
}

// Try to parse string as JSON, return null if not valid JSON
function tryParseJson(content: unknown): { data: unknown; str: string } | null {
  try {
    const parsed = typeof content === 'string' ? JSON.parse(content) : content;
    return { data: parsed, str: JSON.stringify(parsed, null, 2) };
  } catch {
    return null;
  }
}

// Collapsed tool response with JSON rendering + copy
function ToolResponseContent({ step }: { step: AgentStep }) {
  const [isOpen, setIsOpen] = useState(false);
  const parsed = tryParseJson(step.content);
  const copyStr = parsed?.str || step.content || '';

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger className="text-[10px] text-muted-foreground hover:text-foreground flex items-center gap-1">
        <ChevronDown className={cn('h-3 w-3 transition-transform', !isOpen && '-rotate-90')} />
        Response
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-1 rounded-md bg-muted/30 p-3 max-h-[300px] overflow-y-auto relative">
          <div className="absolute top-1 right-1">
            <CopyButton content={copyStr} className="opacity-60 hover:opacity-100" />
          </div>
          {parsed ? <JsonTreeView data={parsed.data} /> : (
            <p className="text-xs text-foreground whitespace-pre-wrap break-words">{step.content}</p>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

// Reasoning step with tool call badges and collapsible inputs
function ReasoningContent({ step }: { step: AgentStep }) {
  const toolCalls = step.tool_calls && step.tool_calls.length > 0 ? step.tool_calls : null;

  return (
    <>
      {step.content && (
        <p className="text-xs text-foreground whitespace-pre-wrap break-words">{step.content}</p>
      )}
      {toolCalls && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {toolCalls.map((tc, i) => (
            <Badge key={i} variant="secondary" className="text-[10px] font-mono gap-1">
              <Wrench className="w-2.5 h-2.5" />
              {tc.tool}
            </Badge>
          ))}
        </div>
      )}
      {toolCalls && (
        <Collapsible>
          <CollapsibleTrigger className="text-[10px] text-muted-foreground hover:text-foreground mt-1 flex items-center gap-1">
            <ChevronDown className="h-3 w-3 -rotate-90" />
            Tool inputs
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-1 rounded-md bg-muted/30 p-3 max-h-[300px] overflow-y-auto">
              {toolCalls.map((tc, i) => {
                const inputStr = typeof tc.input === 'string' ? tc.input : JSON.stringify(tc.input, null, 2);
                return (
                  <div key={i} className="mb-2 last:mb-0">
                    <div className="flex items-center justify-between mb-0.5">
                      <p className="text-[10px] font-mono font-medium text-muted-foreground">{tc.tool}</p>
                      <CopyButton content={inputStr} className="opacity-60 hover:opacity-100" />
                    </div>
                    <JsonTreeView data={tc.input} />
                  </div>
                );
              })}
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}
    </>
  );
}

// Agent step item
function AgentStepItem({ step, index }: { step: AgentStep; index: number }) {
  const isReasoning = step.type === 'reasoning';

  return (
    <div className="flex gap-2">
      <div className="flex flex-col items-center shrink-0">
        <div className={cn(
          'w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-medium',
          isReasoning ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400' : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'
        )}>
          {isReasoning ? <Bot className="w-3 h-3" /> : <Wrench className="w-3 h-3" />}
        </div>
        <div className="w-px flex-1 bg-border" />
      </div>
      <div className="flex-1 min-w-0 pb-3 text-left">
        <div className="flex items-center gap-1.5 mb-1">
          <span className="text-xs font-medium text-muted-foreground">
            {index + 1}. {isReasoning ? 'Reasoning' : `Tool: ${step.tool || 'unknown'}`}
          </span>
        </div>
        {step.type === 'tool_response' ? <ToolResponseContent step={step} /> : <ReasoningContent step={step} />}
      </div>
    </div>
  );
}

// Agent steps section
function AgentStepsSection({ steps, iterations }: { steps: AgentStep[]; iterations?: number }) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="rounded-lg border bg-card">
        <CollapsibleTrigger asChild>
          <button className="flex items-center justify-between w-full p-3 text-left hover:bg-muted/50 transition-colors">
            <div className="flex items-center gap-2">
              <ChevronDown className={cn('h-4 w-4 transition-transform', !isOpen && '-rotate-90')} />
              <MessageSquare className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Agent Trace</span>
              {iterations != null && (
                <Badge variant="secondary" className="text-[10px]">
                  {iterations} iteration{iterations !== 1 ? 's' : ''}
                </Badge>
              )}
            </div>
            <span className="text-xs text-muted-foreground">{steps.length} steps</span>
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="px-3 pb-3 pt-0">
            {steps.map((step, i) => (
              <AgentStepItem key={i} step={step} index={i} />
            ))}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

// Main component
export function NodeDetailPanel({ selectedNode, selectedIndex, executionGraph, workflowId, triggerInfo }: NodeDetailPanelProps) {
  const isSyntheticTrigger = selectedNode.is_synthetic && selectedNode.node_type === 'trigger';

  return (
    <div className="flex flex-col h-full min-w-0 overflow-hidden bg-background">
      {/* Scrollable content */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {isSyntheticTrigger && triggerInfo ? (
            <TriggerDetailView triggerInfo={triggerInfo} />
          ) : (
            <>
              <NodeHeader node={selectedNode} index={selectedIndex} executionGraph={executionGraph} workflowId={workflowId} />
              <DataSection title="Input" data={selectedNode.inputs} />
              <DataSection title="Output" data={selectedNode.output} />
              {selectedNode.node_type === 'agent' && selectedNode.steps && selectedNode.steps.length > 0 && (
                <AgentStepsSection steps={selectedNode.steps} iterations={selectedNode.iterations} />
              )}
              {selectedNode.node_type === 'agent' && selectedNode.prompt && (
                <DataSection title="Prompt" data={selectedNode.prompt} defaultOpen={false} />
              )}
              <ArtifactList artifacts={selectedNode.artifacts} />
            </>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
