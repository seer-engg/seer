import { useAuthStatus } from '@/hooks/useAuthProvider';
import { useQuery } from '@tanstack/react-query';

import type { ToolConnectionStatus } from '@/lib/api-client';
import { toolKeys } from '@/lib/query-keys';
import { listToolConnectionStatus } from '@/lib/tools-api';

export function useToolStatusQuery() {
  const { isLoaded, isSignedIn } = useAuthStatus();

  return useQuery<ToolConnectionStatus[]>({
    queryKey: toolKeys.status(),
    queryFn: listToolConnectionStatus,
    enabled: isLoaded && isSignedIn,
    staleTime: 60 * 1000,
  });
}
