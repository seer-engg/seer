import { useCallback } from 'react';

import { getResourceStatus } from '@/lib/api-client';

export function useCheckResourceValid() {
  return useCallback(async (resourceId: string | number) => {
    try {
      const result = await getResourceStatus(resourceId);
      return result.exists && result.status === 'active';
    } catch (error) {
      console.error('Error checking resource status:', error);
      return false;
    }
  }, []);
}
