import { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, AlertCircle, UserCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { backendApiClient } from '@/lib/api-client';
import { cancelWorkflowRun } from '@/lib/workflows-api';
import { runHistoryKeys } from '@/lib/query-keys';
import { ExecutionTraceHeader } from '@/components/workflows/executions/ExecutionTraceHeader';
import { MetadataTable } from '@/components/workflows/executions/TraceOverview';
import { WaterfallTimeline } from '@/components/workflows/executions/WaterfallTimeline';
import { NodeDetailPanel } from '@/components/workflows/executions/NodeDetailPanel';
import { getNodeDisplayName } from '@/components/workflows/executions/timing-utils';
import type { RunHistoryResponse, RunHistoryEntry, WorkflowNodeTrace } from '@/components/workflows/executions/types';
import { FixWithAI } from '@/components/ui/fix-with-ai';

interface LocationState {
  workflowId?: string | null;
  workflowName?: string | null;
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center py-12">
      <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
    </div>
  );
}

function ErrorState({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 space-y-4">
      <div className="flex items-center gap-2 text-destructive">
        <AlertCircle className="w-5 h-5" />
        <p className="text-sm">Failed to load execution trace</p>
      </div>
      <Button variant="outline" size="sm" onClick={onRetry}>
        Retry
      </Button>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex items-center justify-center py-12">
      <p className="text-sm text-muted-foreground">No trace data available</p>
    </div>
  );
}

interface HITLBannerProps {
  runId: string;
  status: string;
  workflowId?: string | null;
  workflowName?: string | null;
}

function HITLBanner({ runId, status, workflowId, workflowName }: HITLBannerProps) {
  const navigate = useNavigate();

  if (status !== 'interrupted') return null;

  return (
    <div className="flex items-center justify-between gap-4 p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg mb-6">
      <div className="flex items-center gap-3 text-amber-600 dark:text-amber-400">
        <UserCheck className="w-5 h-5 shrink-0" />
        <div>
          <p className="text-sm font-medium">Awaiting human input</p>
          <p className="text-xs text-amber-600/80 dark:text-amber-400/80">
            This workflow is paused and waiting for your response.
          </p>
        </div>
      </div>
      <Button
        size="sm"
        variant="outline"
        className="border-amber-500/50 text-amber-600 dark:text-amber-400 hover:bg-amber-500/10 shrink-0"
        onClick={() =>
          navigate(`/interrupts/${runId}`, { state: { workflowId, workflowName } })
        }
      >
        Respond
      </Button>
    </div>
  );
}

interface TraceContentProps {
  entry: RunHistoryEntry;
  runId: string;
  workflowId?: string | null;
  workflowName?: string | null;
  onCancel?: () => void;
}

const SYNTHETIC_TRIGGER_ID = '__trigger_init__';

/** Returns the selected WorkflowNodeTrace and its display index, handling the synthetic trigger node. */
function resolveSelection(
  nodes: WorkflowNodeTrace[] | undefined,
  selectedIndex: number | null
): { node: WorkflowNodeTrace | null | undefined; index: number } {
  if (selectedIndex === -1) {
    return { node: { node_id: SYNTHETIC_TRIGGER_ID, node_type: 'trigger', is_synthetic: true }, index: 0 };
  }
  if (selectedIndex === null || !nodes?.[selectedIndex]) return { node: null, index: -1 };
  return { node: nodes[selectedIndex], index: selectedIndex };
}

function RunErrorBanner({ error, workflowId }: { error: string; workflowId?: string | null }) {
  return (
    <div className="py-8 space-y-4">
      <div className="rounded-md border border-bug/20 bg-bug/5 p-3 space-y-1">
        <div className="flex items-center gap-2 text-sm font-medium text-bug">
          <AlertCircle className="h-4 w-4" />
          <span>Run Error</span>
          <FixWithAI error={error} workflowId={workflowId ?? undefined} className="ml-auto" />
        </div>
        <p className="text-xs text-muted-foreground font-mono break-all">{error}</p>
      </div>
      <p className="text-sm text-muted-foreground text-center">No execution nodes available</p>
    </div>
  );
}

interface IOTabProps {
  entry: RunHistoryEntry;
  selectedNode: WorkflowNodeTrace | null | undefined;
  selectedIndex: number;
  selectedNodeIndex: number | null;
  onSelectNode: (i: number | null) => void;
  workflowId?: string | null;
}

function IOTab({ entry, selectedNode, selectedIndex, selectedNodeIndex, onSelectNode, workflowId }: IOTabProps) {
  const hasNodes = entry.nodes && entry.nodes.length > 0;

  if (!hasNodes || !selectedNode || selectedIndex === -1) {
    if (entry.error) return <RunErrorBanner error={entry.error} workflowId={workflowId} />;
    return <p className="text-sm text-muted-foreground text-center py-8">No execution nodes available</p>;
  }

  return (
    <div className="space-y-4">
      {entry.nodes && entry.nodes.length > 1 && (
        <div className="flex gap-2 flex-wrap">
          {entry.trigger && (
            <Button variant={selectedNodeIndex === -1 ? 'default' : 'outline'} size="sm" onClick={() => onSelectNode(-1)}>
              Trigger
            </Button>
          )}
          {entry.nodes.map((n, i) => (
            <Button key={i} variant={selectedNodeIndex === i ? 'default' : 'outline'} size="sm" onClick={() => onSelectNode(i)}>
              {getNodeDisplayName(n.node_id, n.node_type, entry.execution_graph)}
            </Button>
          ))}
        </div>
      )}
      <div className="rounded-lg border">
        <NodeDetailPanel
          selectedNode={selectedNode}
          selectedIndex={selectedIndex}
          executionGraph={entry.execution_graph}
          workflowId={workflowId ?? undefined}
          triggerInfo={entry.trigger ?? undefined}
        />
      </div>
    </div>
  );
}

function TraceContent({ entry, runId, workflowId, workflowName, onCancel }: TraceContentProps) {
  const [selectedNodeIndex, setSelectedNodeIndex] = useState<number | null>(
    entry.nodes?.length ? 0 : null
  );

  useEffect(() => {
    if (
      entry.nodes?.length &&
      selectedNodeIndex !== -1 &&
      (selectedNodeIndex === null || selectedNodeIndex >= entry.nodes.length)
    ) {
      setSelectedNodeIndex(0);
    }
  }, [entry.nodes, selectedNodeIndex]);

  const { node: selectedNode, index: selectedIndex } = resolveSelection(entry.nodes, selectedNodeIndex);
  const hasNodes = entry.nodes && entry.nodes.length > 0;

  return (
    <div className="space-y-6">
      <HITLBanner runId={runId} status={entry.status} workflowId={workflowId} workflowName={workflowName} />
      <Tabs defaultValue="io" className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-3">
          <TabsTrigger value="io">Input / Output</TabsTrigger>
          <TabsTrigger value="waterfall">Waterfall</TabsTrigger>
          <TabsTrigger value="metadata">Metadata</TabsTrigger>
        </TabsList>
        <TabsContent value="io" className="mt-6">
          <IOTab
            entry={entry}
            selectedNode={selectedNode}
            selectedIndex={selectedIndex}
            selectedNodeIndex={selectedNodeIndex}
            onSelectNode={setSelectedNodeIndex}
            workflowId={workflowId}
          />
        </TabsContent>
        <TabsContent value="waterfall" className="mt-6">
          {hasNodes && entry.nodes ? (
            <WaterfallTimeline
              nodes={entry.nodes}
              startTime={entry.started_at}
              endTime={entry.finished_at}
              executionGraph={entry.execution_graph}
              selectedNodeIndex={selectedNodeIndex}
              onSelectNode={setSelectedNodeIndex}
              triggerInfo={entry.trigger ?? undefined}
            />
          ) : (
            <p className="text-sm text-muted-foreground text-center py-8">No execution nodes available</p>
          )}
        </TabsContent>
        <TabsContent value="metadata" className="mt-6">
          <MetadataTable entry={entry} onCancel={onCancel} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

export function ExecutionTrace() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();

  const state = location.state as LocationState | null;
  const workflowId = state?.workflowId;
  const workflowName = state?.workflowName;

  const { data, isLoading, error, refetch } = useQuery<RunHistoryResponse>({
    queryKey: runHistoryKeys.detail(runId),
    queryFn: async () => {
      return backendApiClient.request<RunHistoryResponse>(
        `/api/v1/runs/${runId}/history`,
        { method: 'GET' }
      );
    },
    enabled: !!runId,
    refetchInterval: (query) => {
      const response = query.state.data;
      const entry = response?.history?.[0];
      // Keep polling for active statuses including 'interrupted' to catch status changes
      const activeStatuses = ['running', 'queued', 'interrupted'];
      return entry?.status && activeStatuses.includes(entry.status) ? 3000 : false;
    },
  });

  const entry = data?.history?.[0];

  const handleCancel = async () => {
    if (!runId) return;
    await cancelWorkflowRun(runId);
    queryClient.invalidateQueries({ queryKey: runHistoryKeys.detail(runId) });
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      <ExecutionTraceHeader
        runId={runId || ''}
        workflowId={workflowId}
        workflowName={workflowName}
        onBack={() => navigate(-1)}
      />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto p-6 space-y-6">
          {isLoading && <LoadingState />}
          {error && <ErrorState onRetry={refetch} />}
          {!isLoading && !error && !entry && <EmptyState />}
          {!isLoading && !error && entry && runId && <TraceContent entry={entry} runId={runId} workflowId={workflowId} workflowName={workflowName} onCancel={handleCancel} />}
        </div>
      </main>
    </div>
  );
}
