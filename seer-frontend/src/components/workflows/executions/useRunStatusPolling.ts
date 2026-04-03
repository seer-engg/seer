import { useEffect, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import { backendApiClient } from '@/lib/api-client';
import { workflowRunKeys } from '@/lib/query-keys';

import type {
  RunStatus,
  RunStatusResponse,
  WorkflowRunListResponse,
  WorkflowRunSummary,
} from './types';

const ACTIVE_STATUSES: RunStatus[] = ['queued', 'running', 'interrupted'];

interface UseRunStatusPollingOptions {
  workflowId?: string | null;
  runs: WorkflowRunSummary[];
  intervalMs?: number;
}

/**
 * Polls active run statuses and updates the query cache.
 *
 * Uses a ref for `runs` to avoid a feedback loop: setQueryData creates a new
 * runs array → dependency changes → effect restarts → immediate re-poll.
 * With the ref, the effect only restarts when workflowId or intervalMs change.
 */
export function useRunStatusPolling({
  workflowId,
  runs,
  intervalMs = 3000,
}: UseRunStatusPollingOptions) {
  const queryClient = useQueryClient();
  const runsRef = useRef(runs);
  runsRef.current = runs;

  useEffect(() => {
    if (!workflowId) {
      return;
    }

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    const pollOnce = async () => {
      const activeRunIds = runsRef.current
        .filter((run) => ACTIVE_STATUSES.includes(run.status))
        .map((run) => run.run_id);

      if (activeRunIds.length === 0) {
        // No active runs — check again later in case new runs start
        if (!cancelled) {
          timeoutId = setTimeout(pollOnce, intervalMs);
        }
        return;
      }

      try {
        const responses = await Promise.all(
          activeRunIds.map((runId) =>
            backendApiClient.request<RunStatusResponse>(`/api/v1/runs/${runId}`, {
              method: 'GET',
            })
          )
        );

        if (cancelled || responses.length === 0) {
          return;
        }

        const updates = new Map(responses.map((run) => [run.run_id, run]));

        queryClient.setQueryData<WorkflowRunListResponse | undefined>(
          workflowRunKeys.list(workflowId),
          (previous) => {
            if (!previous) {
              return previous;
            }

            const nextRuns = previous.runs.map((run) => {
              const latest = updates.get(run.run_id);
              if (!latest) {
                return run;
              }

              return {
                ...run,
                status: latest.status,
                started_at: latest.started_at ?? run.started_at,
                finished_at: latest.finished_at ?? run.finished_at,
                error: latest.last_error ?? run.error,
              };
            });

            return {
              ...previous,
              runs: nextRuns,
            };
          }
        );
      } catch (error) {
        console.error('Failed to poll run status', error);
      } finally {
        if (!cancelled) {
          timeoutId = setTimeout(pollOnce, intervalMs);
        }
      }
    };

    pollOnce();

    return () => {
      cancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [workflowId, intervalMs, queryClient]);
}
