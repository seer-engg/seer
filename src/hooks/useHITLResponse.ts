/**
 * useHITLResponse Hook
 *
 * Manages fetching and submitting HITL (Human-in-the-Loop) interrupt responses.
 * Automatically fetches interrupt data when a run is in 'interrupted' status.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getRunInterrupt, resumeRun } from '@/lib/hitl-api';
import { runHistoryKeys, runInterruptKeys, workflowRunKeys } from '@/lib/query-keys';
import type { HitlResumePayload } from '@/components/workflows/executions/types';

interface UseHITLResponseOptions {
  /** The ID of the run to check for interrupts */
  runId: string | undefined;
  /** The current status of the run */
  status: string | undefined;
}

/**
 * Hook for managing HITL interrupt responses.
 *
 * @example
 * ```tsx
 * const { interruptData, submitResponse, isSubmitting } = useHITLResponse({
 *   runId: run.id,
 *   status: run.status,
 * });
 *
 * if (interruptData) {
 *   // Render HITL response panel
 * }
 * ```
 */
export function useHITLResponse({ runId, status }: UseHITLResponseOptions) {
  const queryClient = useQueryClient();

  // Fetch interrupt data when run is interrupted
  const interruptQuery = useQuery({
    queryKey: runInterruptKeys.detail(runId),
    queryFn: () => getRunInterrupt(runId!),
    enabled: !!runId && status === 'interrupted',
    // Don't retry too aggressively - the interrupt might not exist yet
    retry: 1,
    // Cache for a short time since interrupt data shouldn't change
    staleTime: 30000,
  });

  // Mutation for resuming the run
  const resumeMutation = useMutation({
    mutationFn: (payload: HitlResumePayload) => resumeRun(runId!, payload),
    onSuccess: () => {
      // Invalidate relevant queries to refresh the UI
      queryClient.invalidateQueries({ queryKey: runHistoryKeys.detail(runId) });
      queryClient.invalidateQueries({ queryKey: runInterruptKeys.detail(runId) });
      queryClient.invalidateQueries({ queryKey: workflowRunKeys.lists() });
    },
  });

  return {
    /** The interrupt data including display items and input fields */
    interruptData: interruptQuery.data,
    /** Whether the interrupt data is currently loading */
    isLoadingInterrupt: interruptQuery.isLoading,
    /** Error that occurred while fetching interrupt data */
    interruptError: interruptQuery.error,
    /** Function to submit the user's response and resume the workflow */
    submitResponse: resumeMutation.mutate,
    /** Function to submit with async handling */
    submitResponseAsync: resumeMutation.mutateAsync,
    /** Whether the response is currently being submitted */
    isSubmitting: resumeMutation.isPending,
    /** Error that occurred while submitting the response */
    submitError: resumeMutation.error,
    /** Whether the submission was successful */
    isSubmitSuccess: resumeMutation.isSuccess,
  };
}
