import { useTriggerCatalogQuery } from '@/hooks/useTriggerCatalogQuery';
import { useTriggersStore } from '@/stores/triggersStore';
import type { TriggerDescriptor } from '@/types/triggers';
import type { TriggerSpec } from '@/types/workflow-spec';

const EMPTY_TRIGGERS: TriggerSpec[] = [];

export interface UseWorkflowTriggersResult {
  triggers: TriggerDescriptor[];
  workflowTriggers: TriggerSpec[];
  isLoadingTriggers: boolean;
  addTrigger: (workflowId: string, trigger: TriggerSpec) => TriggerSpec;
  updateTrigger: (workflowId: string, triggerKey: string, updates: Partial<TriggerSpec>) => TriggerSpec | null;
  removeTrigger: (workflowId: string, triggerKey: string) => void;
}

export function useWorkflowTriggers(workflowId?: string | null): UseWorkflowTriggersResult {
  const { data: triggerCatalog = [], isLoading: triggerCatalogLoading } = useTriggerCatalogQuery();
  const workflowTriggers = useTriggersStore((state) => state.workflowTriggers);
  const addTriggerToStore = useTriggersStore((state) => state.addTrigger);
  const updateTriggerInStore = useTriggersStore((state) => state.updateTrigger);
  const removeTriggerFromStore = useTriggersStore((state) => state.removeTrigger);

  const workflowTriggerList = workflowId ? workflowTriggers.get(workflowId) ?? EMPTY_TRIGGERS : EMPTY_TRIGGERS;

  return {
    triggers: triggerCatalog,
    workflowTriggers: workflowTriggerList,
    isLoadingTriggers: triggerCatalogLoading,
    addTrigger: addTriggerToStore,
    updateTrigger: updateTriggerInStore,
    removeTrigger: removeTriggerFromStore,
  };
}
