import { useMemo } from 'react';
import { useAuth } from '@clerk/clerk-react';
import { useQuery } from '@tanstack/react-query';

import type { FunctionBlockSchema } from '@/components/workflows/types';
import { toolKeys } from '@/lib/query-keys';
import { listFunctionBlocks } from '@/lib/tools-api';

const EMPTY_FUNCTION_BLOCKS: FunctionBlockSchema[] = [];

export function useFunctionBlocksQuery() {
  const { isLoaded, isSignedIn } = useAuth();
  const query = useQuery<FunctionBlockSchema[]>({
    queryKey: toolKeys.functionBlocks(),
    queryFn: listFunctionBlocks,
    enabled: isLoaded && isSignedIn,
    staleTime: 5 * 60 * 1000,
  });

  const functionBlocks = useMemo(() => query.data ?? EMPTY_FUNCTION_BLOCKS, [query.data]);
  const functionBlocksMap = useMemo(() => {
    const next = new Map<string, FunctionBlockSchema>();
    functionBlocks.forEach((block) => {
      next.set(block.type, block);
    });
    return next;
  }, [functionBlocks]);

  return {
    ...query,
    functionBlocks,
    functionBlocksMap,
  };
}
