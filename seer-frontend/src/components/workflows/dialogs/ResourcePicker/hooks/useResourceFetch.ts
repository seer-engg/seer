import { useState, useEffect, useMemo } from 'react';
import { useInfiniteQuery } from '@tanstack/react-query';
import { resourceKeys } from '@/lib/query-keys';
import type { ResourceBrowseResponse } from '../types';
import {
  buildQueryParams,
  fetchResourcePage,
  processCustomEndpointResponse,
} from './resourceFetchHelpers';

interface UseResourceFetchConfig {
  baseEndpoint: string | null;
  resourceType: string;
  searchQuery: string;
  currentParentId?: string;
  dependsOnParam?: string;
  valueField: string;
  displayField: string;
  normalizedEndpoint: string | null;
  searchEnabled?: boolean;
  open: boolean;
  hasMissingDependency: boolean;
}

export function useResourceFetch({
  baseEndpoint,
  resourceType,
  searchQuery,
  currentParentId,
  dependsOnParam,
  valueField,
  displayField,
  normalizedEndpoint,
  searchEnabled = true,
  open,
  hasMissingDependency,
}: UseResourceFetchConfig) {
  const [debouncedQuery, setDebouncedQuery] = useState('');

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const shouldDisableFetcher = !baseEndpoint;

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
    error,
    refetch,
  } = useInfiniteQuery({
    queryKey: resourceKeys.browseList({
      currentParentId,
      dependsOnParam,
      normalizedEndpoint,
      resourceType,
      searchQuery: debouncedQuery,
    }),
    queryFn: async ({ pageParam = undefined }) => {
      if (!baseEndpoint) {
        return { items: [], supports_hierarchy: false, supports_search: false };
      }

      const params = buildQueryParams({
        debouncedQuery,
        searchEnabled,
        normalizedEndpoint,
        currentParentId,
        pageParam,
        dependsOnParam,
      });

      const response = await fetchResourcePage({ baseEndpoint, params });

      if (normalizedEndpoint) {
        return processCustomEndpointResponse(response, valueField, displayField, searchEnabled);
      }

      return response as ResourceBrowseResponse;
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_page_token,
    enabled: open && !hasMissingDependency && !shouldDisableFetcher,
  });

  const items = useMemo(() => data?.pages.flatMap(page => page.items) ?? [], [data]);
  const supportsHierarchy = data?.pages[0]?.supports_hierarchy ?? false;
  const supportsSearch = data?.pages[0]?.supports_search ?? searchEnabled ?? true;

  return {
    items,
    isLoading,
    error: error as Error | null,
    hasNextPage: hasNextPage ?? false,
    fetchNextPage,
    isFetchingNextPage,
    supportsHierarchy,
    supportsSearch,
    shouldDisableFetcher,
    isError,
    refetch,
  };
}
