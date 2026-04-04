import { memo } from 'react';
import type { Node as FlowNode, NodeProps } from '@xyflow/react';

import type { WorkflowNodeData, TriggerNodeMeta } from '../../types';
import { TriggerBlockNodeView } from './components/TriggerBlockNodeView';
import { useTriggerConfig } from './triggers/useTriggerConfig';

type WorkflowNode = FlowNode<WorkflowNodeData>;

/**
 * Inner component that uses hooks - only rendered when triggerMeta exists.
 */
function TriggerBlockNodeInner({
  id,
  label,
  triggerKey,
  triggerMeta,
  selected,
  dragging,
}: {
  id: string;
  label: string;
  triggerKey: string;
  triggerMeta: TriggerNodeMeta;
  selected?: boolean;
  dragging?: boolean;
}) {
  const { kind: triggerKind, descriptor, subscription } = useTriggerConfig(id, triggerKey, triggerMeta);

  const viewProps: import('./components/TriggerBlockNodeView').TriggerBlockNodeViewProps = {
    nodeId: id,
    label,
    dragging,
    selected,
    descriptor,
    subscription,
    triggerKey,
    triggerKind,
  } as const;

  return <TriggerBlockNodeView {...viewProps} />;
}

/**
 * Outer component that handles the null case for triggerMeta.
 */
export const TriggerBlockNode = memo(function TriggerBlockNode({ id, data, selected, dragging }: NodeProps<WorkflowNode>) {
  const triggerMeta = data.triggerMeta;
  const triggerKey = (data.config?.triggerKey as string | undefined) ?? triggerMeta?.subscription?.key ?? '';

  if (!triggerMeta || !triggerKey) {
    return (
      <div className="rounded-lg border-2 border-destructive/40 bg-card p-4 text-sm text-muted-foreground">
        Trigger metadata unavailable
      </div>
    );
  }

  return (
    <TriggerBlockNodeInner
      id={id}
      label={data.label}
      triggerKey={triggerKey}
      triggerMeta={triggerMeta}
      selected={selected}
      dragging={dragging}
    />
  );
});
