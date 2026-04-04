import { useEffect } from 'react';
import { Node } from '@xyflow/react';
import { WorkflowNodeData, WorkflowNodeUpdateOptions } from '../../types';
import { ToolBlockConfig } from '@/components/workflows/block-config/types';
import { buildPersistedNodeUpdate } from './persistedNodeConfig';

export interface UseAutoSaveParams {
  enabled: boolean;
  node: Node<WorkflowNodeData> | null;
  originalNodeRef: React.MutableRefObject<Node<WorkflowNodeData> | null>;
  isSavingRef: React.MutableRefObject<boolean>;
  configRef: React.MutableRefObject<ToolBlockConfig>;
  onUpdate: (
    nodeId: string,
    updates: Partial<WorkflowNodeData>,
    options?: WorkflowNodeUpdateOptions,
  ) => Promise<void> | void;
}

export function useAutoSave({
  enabled,
  node,
  originalNodeRef,
  isSavingRef,
  configRef,
  onUpdate,
}: UseAutoSaveParams): void {
  useEffect(() => {
    if (!enabled) return;

    return () => {
      // We intentionally capture refs in cleanup to get the latest values
      // eslint-disable-next-line react-hooks/exhaustive-deps
      const originalNode = originalNodeRef.current;
      // eslint-disable-next-line react-hooks/exhaustive-deps
      const currentConfig = configRef.current;
      const isSaving = isSavingRef.current;

      // Auto-save when component unmounts (modal closes)
      if (node && !isSaving) {
        if (!originalNode) return;

        const updatePayload = buildPersistedNodeUpdate({
          originalConfig: (originalNode.data.config || {}) as ToolBlockConfig,
          currentConfig,
        });

        if (updatePayload) {
          isSavingRef.current = true;
          void onUpdate(originalNode.id, updatePayload);
          // Reset flag after a delay to allow save to complete
          setTimeout(() => {
            isSavingRef.current = false;
          }, 1000);
        }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, node?.id, onUpdate]);
}
