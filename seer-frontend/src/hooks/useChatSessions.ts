import { useInfiniteQuery } from '@tanstack/react-query';

import { backendApiClient } from '@/lib/api-client';
import { chatKeys } from '@/lib/query-keys';

import type { ChatSession } from '../components/workflows/buildtypes';

export function useChatSessions(workflowId: string | null) {
  return useInfiniteQuery<ChatSession[]>({
    queryKey: chatKeys.sessionsByWorkflow(workflowId),
    queryFn: async ({ pageParam = 0 }) => {
      if (!workflowId) return [];
      const response = await backendApiClient.request<ChatSession[]>(
        `/api/nexus/${workflowId}/chat/sessions?offset=${pageParam}&limit=50`,
        { method: 'GET' },
      );
      return response;
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage, _allPages, lastPageParam) => {
      if (lastPage.length < 50) {
        return undefined;
      }
      return (lastPageParam as number) + 50;
    },
    enabled: !!workflowId,
  });
}
