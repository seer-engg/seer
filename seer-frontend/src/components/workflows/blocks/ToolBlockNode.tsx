/**
 * Tool Block Node Component
 *
 * Displays integration tool blocks with OAuth connection status.
 * Shows connect button for tools requiring authorization.
 */
import { memo, useCallback } from 'react';
import { Handle, NodeProps, type Node as FlowNode } from '@xyflow/react';
import { cn } from '@/lib/utils';
import { WorkflowNodeData } from '../types';
import { useToolIntegration } from '@/hooks/useToolIntegration';
import { useToolIntegrationStatus } from '@/hooks/useToolIntegrationStatus';
import { useWorkflowSave } from '@/hooks/useWorkflowSave';
import { ToolBlockNodeContent } from './ToolBlockNodeContent';
import { OutputEdgeButton } from '../canvas/OutputEdgeButton';
import { NodeDeleteButton } from '../canvas/NodeDeleteButton';
import { getHandlePositionProps, useHandleOrientation } from '../canvas/hooks/useHandleOrientation';
import { NodeErrorIndicator } from './NodeErrorIndicator';
import { useWorkflowCanvasContext } from '../canvas/workflow-canvas-context';

type WorkflowNode = FlowNode<WorkflowNodeData>;

function getToolBorderClass(hasError: boolean, selected: boolean, needsAuth: boolean): string {
  if (hasError) return 'border-bug shadow-lg ring-2 ring-bug/30 ring-offset-2 bg-card';
  if (selected) return 'border-primary shadow-lg ring-2 ring-primary ring-offset-2';
  const base = 'border-border bg-card hover:border-primary/50';
  return needsAuth ? `${base} border-amber-500/50` : base;
}

export const ToolBlockNode = memo(function ToolBlockNode(
  props: NodeProps<WorkflowNode>
) {
  const { data, selected, id, dragging } = props;
  const { readOnly } = useWorkflowCanvasContext();

  const orientation = useHandleOrientation();
  const inputHandlePosition = getHandlePositionProps(orientation, 'input');
  const outputHandlePosition = getHandlePositionProps(orientation, 'output');

  const toolName = data.config?.tool_name || data.config?.toolName || '';
  const connectionId = data.config?.connection_id as number | undefined;
  const { status, isLoading, initiateAuth } = useToolIntegration(toolName, connectionId);
  const { saveWorkflow, hasWorkflow } = useWorkflowSave();

  const handleConnect = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (hasWorkflow) await saveWorkflow();
    const redirectUrl = await initiateAuth();
    if (redirectUrl) window.location.href = redirectUrl;
  }, [initiateAuth, hasWorkflow, saveWorkflow]);

  const { icon, statusBadge, needsAuth } = useToolIntegrationStatus(toolName, status, isLoading);
  const hasError = !!data.error;
  const handleClass = '!w-3 !h-3 !bg-border !border-2 !border-background';

  return (
    <div
      className={cn(
        'group relative px-2 py-2 rounded-lg border-2 min-w-[100px] max-w-[240px] transition-[border,shadow,ring] duration-200 cursor-pointer select-none inline-block',
        getToolBorderClass(hasError, selected, needsAuth),
      )}
    >
      {hasError && data.error && <NodeErrorIndicator error={data.error} />}
      {!readOnly && <NodeDeleteButton nodeId={id} />}

      <Handle type="target" position={inputHandlePosition.position} id="input"
        style={{ position: 'absolute', ...inputHandlePosition.style }} className={handleClass} />

      <ToolBlockNodeContent icon={icon} isLoading={isLoading} needsAuth={needsAuth}
        label={data.label} statusBadge={statusBadge} handleConnect={handleConnect} />

      <Handle type="source" position={outputHandlePosition.position} id="output"
        style={{ position: 'absolute', ...outputHandlePosition.style }} className={handleClass} />

      {!readOnly && !dragging && <OutputEdgeButton nodeId={id} nodeType="tool" />}
    </div>
  );
});
