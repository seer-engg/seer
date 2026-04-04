import { useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import { workflowKeys } from '@/lib/query-keys';
import { useWorkflowVersionsQuery } from './useWorkflowQueries';

export function useWorkflowVersions(workflowId: string | null) {
  const queryClient = useQueryClient();
  const query = useWorkflowVersionsQuery(workflowId);

  const refetch = useCallback(() => {
    if (!workflowId) {
      return Promise.resolve(undefined);
    }

    return query.refetch();
  }, [workflowId, query]);

  const invalidate = useCallback(() => {
    if (!workflowId) {
      return;
    }

    void queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(workflowId) });
  }, [workflowId, queryClient]);

  return {
    versionsResponse: query.versionsResponse,
    versions: query.versions,
    latestVersionId: query.latestVersionId,
    publishedVersionId: query.publishedVersionId,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    refetch,
    invalidate,
  };
}
