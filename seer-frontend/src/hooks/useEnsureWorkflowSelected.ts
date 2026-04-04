import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

interface UseEnsureWorkflowSelectedParams {
  urlWorkflowId: string | undefined;
  workflows: Array<{ workflow_id: string; name: string }>;
  workflowsLoaded: boolean;
  isLoading: boolean;
  workflowListError: unknown;
}

/**
 * Custom hook that ensures a workflow is always selected when on the /workflows page.
 *
 * Behavior:
 * - If workflows exist but none is selected → navigate to the first workflow
 * - If no workflows exist (new user) → navigate to discovery page (/)
 * - Uses guards to prevent race conditions and infinite loops
 *
 * This hook runs after workflows are loaded and only when the URL doesn't contain a workflow ID.
 */
export function useEnsureWorkflowSelected({
  urlWorkflowId,
  workflows,
  workflowsLoaded,
  isLoading,
  workflowListError,
}: UseEnsureWorkflowSelectedParams) {
  const navigate = useNavigate();

  useEffect(() => {
    // Guard 1: Wait for workflows to finish loading
    if (!workflowsLoaded || isLoading) {
      return;
    }

    if (workflowListError) {
      return;
    }

    // Guard 2: Only run on /workflows route (no workflowId in URL)
    if (urlWorkflowId) {
      return;
    }

    // Case 1: Workflows exist → navigate to the first one (topmost)
    if (workflows.length > 0) {
      const firstWorkflow = workflows[0];
      navigate(`/workflows/${firstWorkflow.workflow_id}`, { replace: true });
      return;
    }

    // Case 2: No workflows exist → navigate to discovery page
    navigate('/', { replace: true });
  }, [workflowListError, workflowsLoaded, isLoading, urlWorkflowId, workflows, navigate]);
}
