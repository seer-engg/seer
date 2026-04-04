import { useState, useEffect, useMemo } from 'react';
import { useInfiniteQuery } from '@tanstack/react-query';
import { backendApiClient } from '@/lib/api-client';
import { resourceKeys } from '@/lib/query-keys';
import type { ResourceBrowseResponse } from '@/lib/api-client';

export function useSupabaseProjects(enabled: boolean) {
  const [bindingSearch, setBindingSearch] = useState('');
  const [bindingDebouncedSearch, setBindingDebouncedSearch] = useState('');

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setBindingDebouncedSearch(bindingSearch);
    }, 300);
    return () => clearTimeout(timer);
  }, [bindingSearch]);

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
    error,
  } = useInfiniteQuery({
    queryKey: resourceKeys.supabaseProjects(bindingDebouncedSearch),
    queryFn: async ({ pageParam = undefined }) => {
      const params = new URLSearchParams();
      if (bindingDebouncedSearch) params.set('q', bindingDebouncedSearch);
      if (pageParam) params.set('page_token', pageParam);
      params.set('page_size', '50');
      const endpoint = `/api/integrations/resources/supabase_mgmt/supabase_project?${params.toString()}`;
      return backendApiClient.request<ResourceBrowseResponse>(endpoint, { method: 'GET' });
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_page_token,
    enabled,
  });

  const items = useMemo(() => data?.pages.flatMap(page => page.items) ?? [], [data]);

  return {
    bindingSearch,
    setBindingSearch,
    bindingDebouncedSearch,
    items,
    fetchNextPage,
    hasNextPage: hasNextPage ?? false,
    isFetchingNextPage,
    isLoading,
    isError,
    error,
  };
}
