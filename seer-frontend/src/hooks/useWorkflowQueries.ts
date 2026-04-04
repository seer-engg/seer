import { useMemo } from 'react';
import { useAuthStatus } from '@/hooks/useAuthProvider';
import { useQuery } from '@tanstack/react-query';

import { workflowKeys } from '@/lib/query-keys';
import {
  getWorkflowDetail,
  listWorkflowVersions,
  listWorkflows,
  toWorkflowModel,
  type WorkflowListItem,
  type WorkflowModel,
} from '@/lib/workflows-api';

function useWorkflowQueryEnabled() {
  const { isLoaded, isSignedIn } = useAuthStatus();
  return isLoaded && isSignedIn;
}

export function useWorkflowListQuery() {
  const isQueryEnabled = useWorkflowQueryEnabled();

  return useQuery<WorkflowListItem[]>({
    queryKey: workflowKeys.list(),
    queryFn: listWorkflows,
    enabled: isQueryEnabled,
  });
}

export function useWorkflowDetailQuery(workflowId: string | null | undefined) {
  const isQueryEnabled = useWorkflowQueryEnabled();

  return useQuery<WorkflowModel>({
    queryKey: workflowKeys.detail(workflowId),
    queryFn: async () => {
      if (!workflowId) {
        throw new Error('Workflow ID is required');
      }

      const response = await getWorkflowDetail(workflowId);
      return toWorkflowModel(response);
    },
    enabled: isQueryEnabled && Boolean(workflowId),
  });
}

export function useWorkflowVersionsQuery(workflowId: string | null | undefined) {
  const isQueryEnabled = useWorkflowQueryEnabled();

  const query = useQuery({
    queryKey: workflowKeys.versionList(workflowId),
    queryFn: async () => {
      if (!workflowId) {
        throw new Error('Workflow ID is required');
      }

      return listWorkflowVersions(workflowId);
    },
    enabled: isQueryEnabled && Boolean(workflowId),
  });

  const versions = useMemo(() => query.data?.versions ?? [], [query.data?.versions]);

  return {
    ...query,
    versionsResponse: query.data ?? null,
    versions,
    latestVersionId: query.data?.latest_version_id ?? null,
    publishedVersionId: query.data?.published_version_id ?? null,
  };
}
