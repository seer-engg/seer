/**
 * Hook for fetching trigger events with infinite query pagination.
 * Follows the useResourceFetch pattern from ResourcePicker.
 *
 * Supports two modes:
 * - Polling triggers (gmail, discord): uses providerConnectionId
 * - Persisted triggers (webhooks, forms): uses subscriptionId
 */

import { useMemo, useEffect } from 'react';
import { useInfiniteQuery, useQueryClient } from '@tanstack/react-query';
import { fetchTriggerEvents } from '@/lib/api-client';
import { triggerKeys } from '@/lib/query-keys';
import type { TriggerEventsResponse } from '@/types/triggers';
import { TRIGGER_BROWSING_MODE } from '../types';

interface UseTriggerEventFetchConfig {
  provider: string;
  triggerKey: string;
  /** Provider connection ID for polling triggers */
  providerConnectionId?: number;
  /** Subscription ID for persisted triggers */
  subscriptionId?: number;
  /** Filter params (e.g., channel_id for Discord) */
  filterParams?: Record<string, unknown>;
  open: boolean;
}

export function useTriggerEventFetch({
  provider,
  triggerKey,
  providerConnectionId,
  subscriptionId,
  filterParams,
  open,
}: UseTriggerEventFetchConfig) {
  const queryClient = useQueryClient();
  const browsingMode = TRIGGER_BROWSING_MODE[triggerKey];
  const queryKey = triggerKeys.eventList({
    filterParams,
    provider,
    providerConnectionId,
    subscriptionId,
    triggerKey,
  });

  // Determine if we have the required ID for fetching
  const hasRequiredId =
    (browsingMode === 'polling' && providerConnectionId !== undefined) ||
    (browsingMode === 'persisted' && subscriptionId !== undefined);

  const shouldDisableFetcher = !provider || !triggerKey || !hasRequiredId;

  // When the dialog closes, reset the cache so the next open always fetches fresh data.
  // Resetting on close (not on open) avoids a race where resetQueries cancels the
  // initial fetch that React Query already started when `enabled` became true.
  useEffect(() => {
    if (!open && !shouldDisableFetcher) {
      queryClient.resetQueries({
        queryKey,
        exact: true,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

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
    queryKey,
    queryFn: async ({ pageParam }): Promise<TriggerEventsResponse> => {
      if (shouldDisableFetcher) {
        return { items: [] };
      }

      return fetchTriggerEvents({
        provider,
        triggerKey,
        providerConnectionId,
        subscriptionId,
        pageToken: pageParam,
        filterParams,
      });
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_page_token,
    enabled: open && !shouldDisableFetcher,
  });

  const items = useMemo(() => data?.pages.flatMap((page) => page.items) ?? [], [data]);

  return {
    items,
    isLoading,
    error: error as Error | null,
    hasNextPage: hasNextPage ?? false,
    fetchNextPage,
    isFetchingNextPage,
    isError,
    refetch,
    shouldDisableFetcher,
    browsingMode,
  };
}
