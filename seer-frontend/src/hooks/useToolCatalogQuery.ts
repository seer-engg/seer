import { useAuthStatus } from '@/hooks/useAuthProvider';
import { useQuery } from '@tanstack/react-query';

import { toolKeys } from '@/lib/query-keys';
import { listToolCatalog } from '@/lib/tools-api';
import type { ToolMetadata } from '@/stores/toolsStore';
import { SHOW_MEETINGS } from '@/lib/feature-flags';

const HIDDEN_TOOL_PREFIXES = SHOW_MEETINGS ? [] : ['meetings_ea_'];

export function useToolCatalogQuery() {
  const { isLoaded, isSignedIn } = useAuthStatus();

  return useQuery<ToolMetadata[]>({
    queryKey: toolKeys.catalog(),
    queryFn: () => listToolCatalog() as Promise<ToolMetadata[]>,
    enabled: isLoaded && isSignedIn,
    staleTime: 5 * 60 * 1000,
    select: (data) =>
      HIDDEN_TOOL_PREFIXES.length
        ? data.filter((t) => !HIDDEN_TOOL_PREFIXES.some((prefix) => t.name.startsWith(prefix)))
        : data,
  });
}
