import { QueryClient } from '@tanstack/react-query';

import { defaultQueryOptions } from './query-config';
import { connectionKeys, toolKeys, triggerKeys } from './query-keys';

export const queryClient = new QueryClient({
  defaultOptions: {
    ...defaultQueryOptions,
    queries: {
      ...defaultQueryOptions.queries,
      refetchOnWindowFocus: false,
      refetchOnMount: true,
      staleTime: 5 * 60 * 1000,
      gcTime: 5 * 60 * 1000,
    },
  },
});

export async function invalidateBuilderCatalogQueries(client: QueryClient = queryClient) {
  await Promise.allSettled([
    client.invalidateQueries({ queryKey: toolKeys.catalog() }),
    client.invalidateQueries({ queryKey: toolKeys.status() }),
    client.invalidateQueries({ queryKey: toolKeys.functionBlocks() }),
    client.invalidateQueries({ queryKey: connectionKeys.list() }),
    client.invalidateQueries({ queryKey: triggerKeys.catalog() }),
  ]);
}
