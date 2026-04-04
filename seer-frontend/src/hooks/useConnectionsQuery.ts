import { useAuthStatus } from '@/hooks/useAuthProvider';
import { useQuery } from '@tanstack/react-query';

import type { ConnectedAccount } from '@/lib/api-client';
import { connectionKeys } from '@/lib/query-keys';
import { listConnections } from '@/lib/tools-api';

export function useConnectionsQuery() {
  const { isLoaded, isSignedIn } = useAuthStatus();

  return useQuery<ConnectedAccount[]>({
    queryKey: connectionKeys.list(),
    queryFn: listConnections,
    enabled: isLoaded && isSignedIn,
    staleTime: 60 * 1000,
  });
}
