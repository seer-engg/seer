import { useEffect } from 'react';
import type { Node } from '@xyflow/react';

import type { WorkflowNodeData } from '@/components/workflows/types';
import type { TriggerSpec } from '@/types/workflow-spec';
import { useTriggersStore } from '@/stores/triggersStore';

// Stable empty array to prevent selector from creating new references
const EMPTY_SUBSCRIPTIONS: TriggerSpec[] = [];

/**
 * Hydrates trigger nodes with their corresponding subscription data from the store.
 *
 * This hook solves the persistence issue where trigger configurations save to the backend
 * but disappear on page refresh. The root cause was that normalizeNodes() creates trigger
 * nodes without subscription data, and subscriptions load separately into the store.
 *
 * This hook bridges that gap by:
 * 1. Monitoring both canvas nodes and loaded subscriptions
 * 2. Finding trigger nodes that need subscription data
 * 3. Matching subscriptions by stable trigger id
 * 4. Updating nodes with the matched subscription
 *
 * @param workflowId - Current workflow ID
 * @param nodes - Canvas nodes from React Flow
 * @param setNodes - Function to update canvas nodes
 */
export function useSubscriptionHydration(
  workflowId: string | null,
  nodes: Node<WorkflowNodeData>[],
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void,
) {
  const triggerSubscriptions = useTriggersStore(
    (state) => (workflowId ? state.workflowTriggers.get(workflowId) ?? EMPTY_SUBSCRIPTIONS : EMPTY_SUBSCRIPTIONS),
  );

  useEffect(() => {
    if (!workflowId || !nodes.length || !triggerSubscriptions.length) {
      return;
    }

    // Check if any trigger nodes need subscription data
    const needsHydration = nodes.some(
      (node) =>
        node.type === 'trigger' &&
        node.data.triggerMeta &&
        !node.data.triggerMeta.subscription &&
        !node.data.triggerMeta.draft,
    );

    if (!needsHydration) {
      return;
    }

    // Track whether we actually modified any nodes
    let hasChanges = false;

    // Map subscriptions to trigger nodes
    const updatedNodes = nodes.map((node) => {
      // Only process trigger nodes
      if (node.type !== 'trigger' || !node.data.triggerMeta) {
        return node;
      }

      const { triggerMeta } = node.data;

      // Skip if already has subscription or is a draft (unsaved trigger)
      if (triggerMeta.subscription || triggerMeta.draft) {
        return node;
      }

      const matchingSubscription = triggerSubscriptions.find(
        (sub) => sub.id === node.id,
      );

      if (!matchingSubscription) {
        return node;
      }

      // Mark that we're making a change
      hasChanges = true;

      // Hydrate node with subscription data
      return {
        ...node,
        data: {
          ...node.data,
          triggerMeta: {
            ...triggerMeta,
            subscription: matchingSubscription,
          },
        },
      };
    });

    // Only call setNodes if we actually hydrated at least one node
    if (hasChanges) {
      setNodes(updatedNodes);
    }
  }, [workflowId, nodes, triggerSubscriptions, setNodes]);
}
