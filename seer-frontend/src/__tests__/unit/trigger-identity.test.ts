import { beforeEach, describe, expect, it } from 'vitest';
import type { Node } from '@xyflow/react';

import type { WorkflowNodeData } from '@/components/workflows/types';
import { syncTriggerNodesToStore } from '@/lib/workflow-graph';
import { useTriggersStore } from '@/stores/triggersStore';
import { resetAllStores } from '@/test/utils/test-utils';
import type { TriggerSpec } from '@/types/workflow-spec';

const buildTrigger = (id: string, connectionId: number | null): TriggerSpec => ({
  id,
  key: 'poll.gmail.email_received',
  mode: 'poll',
  event_schema: {},
  meta: {},
  provider_config: connectionId === null ? null : { provider_connection_id: connectionId },
  ui_meta: {},
});

describe('trigger identity regressions', () => {
  beforeEach(() => {
    resetAllStores();
  });

  it('updates only the trigger instance that matches the provided id', () => {
    const workflowId = 'workflow-1';
    const firstTrigger = buildTrigger('trigger-1', 101);
    const secondTrigger = buildTrigger('trigger-2', 202);

    useTriggersStore.getState().loadTriggersFromSpec(workflowId, [firstTrigger, secondTrigger]);
    useTriggersStore.getState().updateTrigger(workflowId, secondTrigger.id, {
      provider_config: { provider_connection_id: 999 },
    });

    const updatedTriggers = useTriggersStore.getState().getWorkflowTriggers(workflowId);

    expect(updatedTriggers).toHaveLength(2);
    expect(updatedTriggers.find((trigger) => trigger.id === firstTrigger.id)?.provider_config).toEqual({
      provider_connection_id: 101,
    });
    expect(updatedTriggers.find((trigger) => trigger.id === secondTrigger.id)?.provider_config).toEqual({
      provider_connection_id: 999,
    });
  });

  it('syncs trigger node subscriptions by node id instead of shared trigger key', () => {
    const triggerSpecs = [buildTrigger('trigger-1', 101), buildTrigger('trigger-2', 202)];

    const nodes: Node<WorkflowNodeData>[] = triggerSpecs.map((trigger) => ({
      id: trigger.id,
      type: 'trigger',
      position: { x: 0, y: 0 },
      data: {
        type: 'trigger',
        label: trigger.id,
        config: { triggerKey: trigger.key },
        triggerMeta: {
          handlers: {},
        },
      },
    }));

    const syncedNodes = syncTriggerNodesToStore(nodes, triggerSpecs);

    expect(syncedNodes[0].data.triggerMeta?.subscription?.id).toBe('trigger-1');
    expect(syncedNodes[0].data.triggerMeta?.subscription?.provider_config).toEqual({
      provider_connection_id: 101,
    });
    expect(syncedNodes[1].data.triggerMeta?.subscription?.id).toBe('trigger-2');
    expect(syncedNodes[1].data.triggerMeta?.subscription?.provider_config).toEqual({
      provider_connection_id: 202,
    });
  });
});
