import { useEffect, useRef } from 'react';
import { Node } from '@xyflow/react';
import { WorkflowNodeData, WorkflowNodeUpdateOptions } from '../../types';
import { ToolBlockConfig } from '@/components/workflows/block-config/types';
import { buildPersistedNodeUpdate } from './persistedNodeConfig';

// Helper function to trigger live update with debounce
function triggerLiveUpdate(params: {
  node: Node<WorkflowNodeData>;
  liveUpdateDelayMs: number;
  liveUpdateTimeoutRef: React.MutableRefObject<ReturnType<typeof setTimeout> | null>;
  configRef: React.MutableRefObject<ToolBlockConfig>;
  onUpdate: (
    nodeId: string,
    updates: Partial<WorkflowNodeData>,
    options?: WorkflowNodeUpdateOptions,
  ) => Promise<void> | void;
}): void {
  const { node, liveUpdateDelayMs, liveUpdateTimeoutRef, configRef, onUpdate } = params;

  if (liveUpdateTimeoutRef.current) {
    clearTimeout(liveUpdateTimeoutRef.current);
  }

  liveUpdateTimeoutRef.current = setTimeout(() => {
    // Use refs to get latest values when timeout fires
    const latestConfig = configRef.current;
    const updatePayload = buildPersistedNodeUpdate({
      originalConfig: (node.data.config || {}) as ToolBlockConfig,
      currentConfig: latestConfig,
    });

    if (updatePayload) {
      void onUpdate(node.id, updatePayload);
    }
    liveUpdateTimeoutRef.current = null;
  }, liveUpdateDelayMs);
}

export interface UseLiveUpdateParams {
  enabled: boolean;
  delayMs: number;
  node: Node<WorkflowNodeData> | null;
  currentConfig: ToolBlockConfig;
  currentInputRefs: Record<string, string>;
  currentOauthScope: string | undefined;
  configRef: React.MutableRefObject<ToolBlockConfig>;
  onUpdate: (
    nodeId: string,
    updates: Partial<WorkflowNodeData>,
    options?: WorkflowNodeUpdateOptions,
  ) => Promise<void> | void;
}

export function useLiveUpdate({
  enabled,
  delayMs,
  node,
  currentConfig,
  currentInputRefs,
  currentOauthScope,
  configRef,
  onUpdate,
}: UseLiveUpdateParams): void {
  const liveUpdateTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!enabled || !node) {
      return;
    }

    const nodeConfig = node.data.config || {};
    const changesDetected = buildPersistedNodeUpdate({
      originalConfig: nodeConfig as ToolBlockConfig,
      currentConfig,
    });

    if (!changesDetected) {
      return;
    }

    triggerLiveUpdate({
      node,
      liveUpdateDelayMs: delayMs,
      liveUpdateTimeoutRef,
      configRef,
      onUpdate,
    });

    return () => {
      if (liveUpdateTimeoutRef.current) {
        clearTimeout(liveUpdateTimeoutRef.current);
        liveUpdateTimeoutRef.current = null;
      }
    };
    // Use node?.id instead of node to prevent circular updates when store updates cause new node reference
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, delayMs, node?.id, currentConfig, currentInputRefs, currentOauthScope, configRef, onUpdate]);
}
