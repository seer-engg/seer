import { useQuery } from '@tanstack/react-query';

import { triggerKeys } from '@/lib/query-keys';
import { listTriggers } from '@/lib/workflow-triggers';
import type { TriggerDescriptor } from '@/types/triggers';
import { SHOW_MEETINGS } from '@/lib/feature-flags';

const HIDDEN_TRIGGER_PROVIDERS = SHOW_MEETINGS ? [] : ['meetings_ea'];

export function useTriggerCatalogQuery({ enabled = true }: { enabled?: boolean } = {}) {
  return useQuery<TriggerDescriptor[]>({
    queryKey: triggerKeys.catalog(),
    queryFn: listTriggers,
    staleTime: 5 * 60 * 1000,
    enabled,
    select: (data) =>
      HIDDEN_TRIGGER_PROVIDERS.length
        ? data.filter((t) => !HIDDEN_TRIGGER_PROVIDERS.includes(t.provider))
        : data,
  });
}
