import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { Loader2, AlertCircle, UserCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ExecutionTraceHeader } from '@/components/workflows/executions/ExecutionTraceHeader';
import { HITLResponsePanel } from '@/components/workflows/executions/HITLResponsePanel';
import { BrowserHITLPanel } from '@/components/workflows/executions/BrowserHITLPanel';
import { useHITLResponse } from '@/hooks/useHITLResponse';
import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { backendApiClient } from '@/lib/api-client';
import { runHistoryKeys } from '@/lib/query-keys';
import type { RunHistoryResponse, HitlInterruptData } from '@/components/workflows/executions/types';

interface LocationState {
  workflowId?: string | null;
  workflowName?: string | null;
}

interface InterruptContentProps {
  runStatus: string | undefined;
  isLoadingInterrupt: boolean;
  interruptData: HitlInterruptData | undefined;
  isSubmitting: boolean;
  submitError: Error | null;
  onNavigateToTrace: () => void;
  onSubmit: (response: unknown) => void;
}

function InterruptContent({
  runStatus,
  isLoadingInterrupt,
  interruptData,
  isSubmitting,
  submitError,
  onNavigateToTrace,
  onSubmit,
}: InterruptContentProps) {
  const isNotInterrupted = runStatus && runStatus !== 'interrupted';

  if (isNotInterrupted) {
    return (
      <div className="flex flex-col items-center justify-center py-16 space-y-4 text-center">
        <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
          <UserCheck className="w-6 h-6 text-muted-foreground" />
        </div>
        <div className="space-y-1">
          <p className="text-sm font-medium">Interrupt already resolved</p>
          <p className="text-xs text-muted-foreground">
            This run is no longer waiting for input (status: {runStatus}).
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={onNavigateToTrace}>
          View Execution Trace
        </Button>
      </div>
    );
  }

  if (isLoadingInterrupt) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        <span className="ml-2 text-sm text-muted-foreground">Loading interrupt data…</span>
      </div>
    );
  }

  if (!interruptData) {
    return (
      <div className="flex items-center gap-2 text-amber-600 text-sm p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
        <AlertCircle className="w-4 h-4 shrink-0" />
        <span>Workflow is waiting for human input, but interrupt data is not available.</span>
      </div>
    );
  }

  // Browser HITL: show live browser viewer + response form
  if (interruptData.type === 'browser_hitl' && interruptData.session_id) {
    return (
      <BrowserHITLPanel
        interruptData={interruptData}
        onSubmit={onSubmit}
        isSubmitting={isSubmitting}
        error={submitError}
      />
    );
  }

  return (
    <HITLResponsePanel
      interruptData={interruptData}
      onSubmit={onSubmit}
      isSubmitting={isSubmitting}
      error={submitError}
    />
  );
}

export function InterruptResume() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const location = useLocation();

  const state = location.state as LocationState | null;
  const workflowId = state?.workflowId;
  const workflowName = state?.workflowName;

  const { data: historyData } = useQuery<RunHistoryResponse>({
    queryKey: runHistoryKeys.detail(runId),
    queryFn: () =>
      backendApiClient.request<RunHistoryResponse>(`/api/v1/runs/${runId}/history`, {
        method: 'GET',
      }),
    enabled: !!runId,
    refetchInterval: 3000,
  });

  const runStatus = historyData?.history?.[0]?.status;

  const { interruptData, isLoadingInterrupt, submitResponse, isSubmitting, submitError, isSubmitSuccess } =
    useHITLResponse({ runId, status: runStatus ?? 'interrupted' });

  const navigateBack = () =>
    workflowId ? navigate(`/workflows/${workflowId}`) : navigate('/workflows');

  useEffect(() => {
    if (isSubmitSuccess) {
      navigate(`/executions/${runId}`, { replace: true, state: { workflowId, workflowName } });
    }
  }, [isSubmitSuccess, navigate, runId, workflowId, workflowName]);

  return (
    <div className="flex flex-col h-screen bg-background">
      <ExecutionTraceHeader
        runId={runId ?? ''}
        workflowId={workflowId}
        workflowName={workflowName}
        onBack={navigateBack}
      />

      <main className="flex-1 overflow-y-auto">
        <div className={`mx-auto p-6 space-y-6 ${
          interruptData?.type === 'browser_hitl' && interruptData?.session_id
            ? 'max-w-7xl'
            : 'max-w-3xl'
        }`}>
          <InterruptContent
            runStatus={runStatus}
            isLoadingInterrupt={isLoadingInterrupt}
            interruptData={interruptData}
            isSubmitting={isSubmitting}
            submitError={submitError}
            onNavigateToTrace={navigateBack}
            onSubmit={submitResponse}
          />
        </div>
      </main>
    </div>
  );
}
