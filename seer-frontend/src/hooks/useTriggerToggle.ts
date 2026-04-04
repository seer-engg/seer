import { useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  listTriggerSubscriptions,
  updateTriggerSubscriptionEnabled,
} from '@/lib/api-client';
import { triggerKeys } from '@/lib/query-keys';
import { toast } from '@/components/ui/sonner';

/**
 * Hook for toggling trigger subscription enabled state with optimistic updates.
 * Provides isEnabled state, isPending state, and handleToggle callback.
 */
export function useTriggerToggle(subscriptionId: number | null) {
  const queryClient = useQueryClient();

  // Fetch actual enabled state from backend
  const { data: subscriptionData } = useQuery({
    queryKey: triggerKeys.subscriptionEnabledDetail(subscriptionId),
    queryFn: async () => {
      if (!subscriptionId) return null;
      const response = await listTriggerSubscriptions();
      return response.items.find((item) => item.id === subscriptionId) ?? null;
    },
    enabled: !!subscriptionId,
    staleTime: 30_000,
  });

  const isEnabled = subscriptionData?.enabled ?? true;

  // Mutation for toggling with optimistic updates
  const toggleMutation = useMutation({
    mutationFn: async (enabled: boolean) => {
      if (!subscriptionId) throw new Error('No subscription ID');
      await updateTriggerSubscriptionEnabled(subscriptionId, enabled);
      return enabled;
    },
    onMutate: async (enabled) => {
      await queryClient.cancelQueries({
        queryKey: triggerKeys.subscriptionEnabledDetail(subscriptionId),
      });
      const previous = queryClient.getQueryData(
        triggerKeys.subscriptionEnabledDetail(subscriptionId),
      );
      queryClient.setQueryData(
        triggerKeys.subscriptionEnabledDetail(subscriptionId),
        (old: typeof subscriptionData) => (old ? { ...old, enabled } : old)
      );
      return { previous };
    },
    onError: (_err, _enabled, context) => {
      if (context?.previous) {
        queryClient.setQueryData(
          triggerKeys.subscriptionEnabledDetail(subscriptionId),
          context.previous
        );
      }
      toast.error('Failed to update trigger');
    },
    onSuccess: (enabled) => {
      toast.success(enabled ? 'Trigger enabled' : 'Trigger disabled');
      queryClient.invalidateQueries({ queryKey: triggerKeys.subscriptions() });
    },
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: triggerKeys.subscriptionEnabledDetail(subscriptionId),
      });
    },
  });

  const handleToggle = useCallback(
    (checked: boolean) => {
      if (!subscriptionId) {
        toast.error('Trigger not yet saved');
        return;
      }
      toggleMutation.mutate(checked);
    },
    [subscriptionId, toggleMutation]
  );

  return {
    isEnabled,
    isPending: toggleMutation.isPending,
    handleToggle,
  };
}
