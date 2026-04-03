import { useMemo } from 'react';
import type { UnifiedItem } from '@/components/workflows/buildtypes';

/**
 * Filter items by search query
 */
export function useItemSearch(
  items: UnifiedItem[],
  searchQuery: string
): UnifiedItem[] {
  return useMemo(() => {
    if (!searchQuery.trim()) {
      return items;
    }

    const query = searchQuery.toLowerCase();
    return items.filter((item) => item.searchTerms.includes(query));
  }, [items, searchQuery]);
}
