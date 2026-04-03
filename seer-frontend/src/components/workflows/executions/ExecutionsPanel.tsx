/**
 * ExecutionsPanel - List of workflow executions for the sidebar
 */
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Query } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { format } from 'date-fns';
import { CheckCircle, XCircle, Loader2, Clock, AlertCircle, Play, ExternalLink, UserCheck, ScrollText, Square } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { backendApiClient } from '@/lib/api-client';
import { cancelWorkflowRun } from '@/lib/workflows-api';
import { useWorkflowDetailQuery } from '@/hooks/useWorkflowQueries';
import { workflowRunKeys } from '@/lib/query-keys';
import { shouldStopPolling } from '@/lib/error-handler';

import type { RunStatus, WorkflowRunListResponse, WorkflowRunSummary } from './types';

interface ExecutionsPanelProps {
  workflowId: string | null;
  isOpen?: boolean;
}

function getStatusIcon(status: RunStatus) {
  switch (status) {
    case 'succeeded':
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    case 'failed':
      return <XCircle className="w-4 h-4 text-red-500" />;
    case 'running':
      return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
    case 'interrupted':
      return <UserCheck className="w-4 h-4 text-amber-500" />;
    case 'cancelled':
      return <AlertCircle className="w-4 h-4 text-muted-foreground" />;
    case 'queued':
      return <Clock className="w-4 h-4 text-muted-foreground" />;
    default:
      return <Clock className="w-4 h-4 text-muted-foreground" />;
  }
}

function getStatusBadge(status: RunStatus) {
  const variants: Record<RunStatus, 'default' | 'destructive' | 'secondary' | 'outline'> = {
    succeeded: 'default',
    failed: 'destructive',
    running: 'secondary',
    queued: 'secondary',
    interrupted: 'outline',
    cancelled: 'outline',
  };

  // Custom styling for interrupted status
  if (status === 'interrupted') {
    return (
      <Badge variant="outline" className="text-xs border-amber-500/50 text-amber-600 bg-amber-500/10">
        Awaiting Input
      </Badge>
    );
  }

  return (
    <Badge variant={variants[status]} className="text-xs">
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  );
}

function calculateDuration(startedAt?: string | null, finishedAt?: string | null): string | null {
  if (!startedAt || !finishedAt) return null;
  const seconds = Math.round(
    (new Date(finishedAt).getTime() - new Date(startedAt).getTime()) / 1000
  );
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

function NoWorkflowState() {
  return (
    <div className="flex items-center justify-center flex-1 text-muted-foreground">
      <p className="text-sm">Select a workflow to view executions</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center flex-1">
      <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
    </div>
  );
}

function ErrorState() {
  return (
    <div className="flex items-center justify-center flex-1 text-muted-foreground">
      <p className="text-sm">Failed to load executions</p>
    </div>
  );
}

function EmptyExecutionsState() {
  return (
    <div className="flex items-center justify-center flex-1">
      <div className="text-center space-y-3 p-6">
        <div className="w-12 h-12 mx-auto bg-muted rounded-full flex items-center justify-center">
          <Play className="w-5 h-5 text-muted-foreground" />
        </div>
        <div>
          <p className="text-sm font-medium">No executions yet</p>
          <p className="text-xs text-muted-foreground mt-1">
            Run this workflow to see execution history
          </p>
        </div>
      </div>
    </div>
  );
}

interface ExecutionItemProps {
  run: WorkflowRunSummary;
  onLogsClick: () => void;
  onInterruptClick?: () => void;
  onCancelClick?: () => void;
}

function ExecutionItem({ run, onLogsClick, onInterruptClick, onCancelClick }: ExecutionItemProps) {
  const duration = calculateDuration(run.started_at, run.finished_at);
  const isInterrupted = run.status === 'interrupted';
  return (
    <div className="w-full text-left p-3 rounded-md border bg-card transition-colors">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          {getStatusIcon(run.status)}
          {getStatusBadge(run.status)}
        </div>
        <ExternalLink className="w-3.5 h-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
      </div>
      <div className="space-y-1">
        <p className="text-xs font-mono text-muted-foreground truncate">{run.run_id}</p>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{format(new Date(run.created_at), 'MMM d, h:mm a')}</span>
          {duration && (
            <>
              <span>•</span>
              <span>{duration}</span>
            </>
          )}
        </div>
        {run.error && <p className="text-xs text-destructive line-clamp-1">{run.error}</p>}
      </div>
      <div className="flex items-center gap-1 mt-2 pt-2 border-t border-border">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs gap-1.5"
          onClick={onLogsClick}
        >
          <ScrollText className="w-3 h-3" />
          Logs
        </Button>
        {isInterrupted && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs gap-1.5 text-amber-600 dark:text-amber-400 hover:text-amber-600 dark:hover:text-amber-400 hover:bg-amber-500/10"
            onClick={onInterruptClick}
          >
            <UserCheck className="w-3 h-3" />
            Respond
          </Button>
        )}
        {onCancelClick && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs gap-1.5 text-destructive hover:text-destructive hover:bg-destructive/10"
            onClick={onCancelClick}
          >
            <Square className="w-3 h-3" />
            Cancel
          </Button>
        )}
      </div>
    </div>
  );
}

export function ExecutionsPanel({ workflowId, isOpen }: ExecutionsPanelProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: workflow } = useWorkflowDetailQuery(workflowId);

  const wasOpenRef = useRef(false);

  const { data, isLoading, isError, refetch } = useQuery<WorkflowRunListResponse>({
    queryKey: workflowRunKeys.list(workflowId),
    queryFn: async () => {
      return backendApiClient.request<WorkflowRunListResponse>(
        `/api/v1/workflows/${workflowId}/runs`,
        { method: 'GET' }
      );
    },
    enabled: !!workflowId,
    refetchInterval: (
      query: Query<WorkflowRunListResponse, Error, WorkflowRunListResponse, readonly unknown[]>
    ) => {
      // Stop polling on non-retryable errors (400, 402, 500, etc.)
      if (query.state.error && shouldStopPolling(query.state.error)) {
        return false;
      }

      const response = query.state.data;
      const hasActive = response?.runs?.some(
        (run) => run.status === 'running' || run.status === 'queued' || run.status === 'interrupted'
      );
      return hasActive ? 3000 : false;
    },
  });

  // Refetch when the panel is opened (false → true transition)
  useEffect(() => {
    if (isOpen && !wasOpenRef.current && workflowId) {
      refetch();
    }
    wasOpenRef.current = !!isOpen;
  }, [isOpen, workflowId, refetch]);

  const runs: WorkflowRunSummary[] = data?.runs ?? [];

  const navState = { workflowId, workflowName: workflow?.name || 'Workflow' };

  const handleLogsClick = (runId: string) =>
    navigate(`/executions/${runId}`, { state: navState });

  const handleInterruptClick = (runId: string) =>
    navigate(`/interrupts/${runId}`, { state: navState });

  const handleCancelClick = async (runId: string) => {
    await cancelWorkflowRun(runId);
    queryClient.invalidateQueries({ queryKey: workflowRunKeys.list(workflowId) });
  };

  if (!workflowId) return <NoWorkflowState />;
  if (isLoading) return <LoadingState />;
  if (isError) return <ErrorState />;

  return (
    <div className="flex flex-col h-full bg-background">
      {runs.length === 0 ? (
        <EmptyExecutionsState />
      ) : (
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {runs.map((run) => (
              <ExecutionItem
                key={run.run_id}
                run={run}
                onLogsClick={() => handleLogsClick(run.run_id)}
                onCancelClick={['running', 'queued', 'interrupted'].includes(run.status) ? () => handleCancelClick(run.run_id) : undefined}
                onInterruptClick={run.status === 'interrupted' ? () => handleInterruptClick(run.run_id) : undefined}
              />
            ))}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}
