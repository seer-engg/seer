import { useState, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  listTriggerSubscriptions,
  updateTriggerSubscriptionEnabled,
} from '@/lib/api-client';
import { triggerKeys } from '@/lib/query-keys';
import type {
  TriggerSubscriptionListItem,
  TriggerSubscriptionFilters,
} from '@/types/triggers';

export interface UseTriggerSubscriptionsOptions {
  /** Initial filters to apply */
  initialFilters?: TriggerSubscriptionFilters;
}

export function useTriggerSubscriptions(options: UseTriggerSubscriptionsOptions = {}) {
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState<TriggerSubscriptionFilters>(
    options.initialFilters ?? {}
  );

  // Query for fetching subscriptions
  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: triggerKeys.subscriptionList(filters),
    queryFn: () => listTriggerSubscriptions(filters),
  });

  // Memoize subscriptions to avoid recreating array on each render
  const subscriptions = useMemo(() => data?.items ?? [], [data?.items]);

  // Mutation for toggling enabled status with optimistic updates
  const toggleMutation = useMutation({
    mutationFn: async ({
      subscriptionId,
      enabled,
    }: {
      subscriptionId: number;
      enabled: boolean;
    }) => {
      await updateTriggerSubscriptionEnabled(subscriptionId, enabled);
      return { subscriptionId, enabled };
    },
    // Optimistic update
    onMutate: async ({ subscriptionId, enabled }) => {
      const subscriptionListKey = triggerKeys.subscriptionList(filters);

      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: subscriptionListKey });

      // Snapshot the previous value
      const previousData = queryClient.getQueryData(subscriptionListKey);

      // Optimistically update
      queryClient.setQueryData(
        subscriptionListKey,
        (old: { items: TriggerSubscriptionListItem[] } | undefined) => {
          if (!old) return old;
          return {
            ...old,
            items: old.items.map((item) =>
              item.id === subscriptionId ? { ...item, enabled } : item
            ),
          };
        }
      );

      return { previousData };
    },
    // Rollback on error
    onError: (_err, _vars, context) => {
      if (context?.previousData) {
        queryClient.setQueryData(triggerKeys.subscriptionList(filters), context.previousData);
      }
    },
    // Refetch after mutation
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: triggerKeys.subscriptionList(filters) });
      // Also invalidate the per-subscription queries used by canvas trigger nodes
      queryClient.invalidateQueries({ queryKey: triggerKeys.subscriptionEnabled() });
    },
  });

  const toggleEnabled = useCallback(
    (subscriptionId: number, enabled: boolean) => {
      toggleMutation.mutate({ subscriptionId, enabled });
    },
    [toggleMutation]
  );

  // Filter update helpers
  const updateFilter = useCallback(
    <K extends keyof TriggerSubscriptionFilters>(
      key: K,
      value: TriggerSubscriptionFilters[K]
    ) => {
      setFilters((prev) => ({
        ...prev,
        [key]: value || undefined, // Remove empty values
      }));
    },
    []
  );

  const clearFilters = useCallback(() => {
    setFilters({});
  }, []);

  // Unique trigger keys for filter dropdown
  const uniqueTriggerKeys = useMemo(() => {
    const keys = new Set(subscriptions.map((s) => s.trigger_key));
    return Array.from(keys).sort();
  }, [subscriptions]);

  // Unique workflows for filter dropdown
  const uniqueWorkflows = useMemo(() => {
    const workflowMap = new Map<string, string>();
    subscriptions.forEach((s) => {
      if (!workflowMap.has(s.workflow_id)) {
        workflowMap.set(s.workflow_id, s.workflow_title);
      }
    });
    return Array.from(workflowMap.entries()).map(([id, title]) => ({
      id,
      title,
    }));
  }, [subscriptions]);

  return {
    // Data
    subscriptions,
    isLoading,
    isError,
    error,

    // Filters
    filters,
    updateFilter,
    clearFilters,
    uniqueTriggerKeys,
    uniqueWorkflows,

    // Actions
    toggleEnabled,
    isToggling: toggleMutation.isPending,
    refetch,
  };
}
