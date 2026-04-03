/**
 * Base Block Node Component
 *
 * Common functionality for all workflow block nodes.
 */
import { memo } from 'react';
import { Handle, NodeProps, type Node } from '@xyflow/react';
import { Zap } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip';
import { WorkflowNodeData } from '../types';
import { OutputEdgeButton } from '../canvas/OutputEdgeButton';
import { NodeDeleteButton } from '../canvas/NodeDeleteButton';
import { getHandlePositionProps, useHandleOrientation } from '../canvas/hooks/useHandleOrientation';
import { NodeErrorIndicator } from './NodeErrorIndicator';
import { useWorkflowCanvasContext } from '../canvas/workflow-canvas-context';

type CanvasNode = Node<WorkflowNodeData>;

interface BaseBlockNodeProps extends NodeProps<CanvasNode> {
  icon?: React.ReactNode;
  color?: string;
  handles?: { inputs?: string[]; outputs?: string[] };
  children?: React.ReactNode;
  minWidth?: string;
  outputButtons?: React.ReactNode;
  consumesCredits?: boolean;
}

/** AI credits badge shown on nodes that consume AI credits */
function AiCreditsBadge() {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex items-center gap-0.5 rounded-full border px-1 py-0.5 text-[8px] font-medium leading-none bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20 cursor-default shrink-0">
          <Zap className="w-2.5 h-2.5" />
          AI
        </span>
      </TooltipTrigger>
      <TooltipContent side="top">
        <p className="text-xs">This node consumes AI credits</p>
      </TooltipContent>
    </Tooltip>
  );
}

function getNodeBorderClass(hasError: boolean, selected: boolean): string {
  if (hasError) return 'border-bug shadow-lg ring-2 ring-bug/30 ring-offset-2 bg-card';
  if (selected) return 'border-primary shadow-lg ring-2 ring-primary ring-offset-2 bg-card';
  return 'border-border bg-card hover:border-primary/50';
}

const HANDLE_CLASS = '!w-3 !h-3 !bg-border !border-2 !border-background';

function NodeHeader({ icon, color, label, consumesCredits }: { icon?: React.ReactNode; color: string; label: string; consumesCredits?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      {icon && (
        <div className={cn('w-8 h-8 rounded flex items-center justify-center', `bg-${color}/10`)}>
          {icon}
        </div>
      )}
      <div className="flex-1 min-w-0">
        <p className="font-medium text-sm truncate">{label}</p>
      </div>
      {consumesCredits && <AiCreditsBadge />}
    </div>
  );
}

export const BaseBlockNode = memo(function BaseBlockNode({
  id, data, selected, dragging, icon, color = 'primary',
  handles = { inputs: ['input'], outputs: ['output'] },
  children, minWidth = '120px', outputButtons, consumesCredits,
}: BaseBlockNodeProps) {
  const { readOnly } = useWorkflowCanvasContext();
  const { inputs = ['input'], outputs = ['output'] } = handles;
  const hasInputHandle = inputs.length > 0;
  const hasOutputHandle = outputs.length > 0;
  const orientation = useHandleOrientation();
  const inputHandlePosition = getHandlePositionProps(orientation, 'input');
  const outputHandlePosition = getHandlePositionProps(orientation, 'output');
  const hasError = !!data.error;

  return (
    <div
      className={cn(
        'group relative px-2 py-2 rounded-lg border-2 transition-[border,shadow,ring] duration-200 cursor-pointer select-none inline-block',
        getNodeBorderClass(hasError, selected),
      )}
      style={{ minWidth, maxWidth: '240px', width: 'fit-content' }}
    >
      {data.error && <NodeErrorIndicator error={data.error} />}
      {!readOnly && <NodeDeleteButton nodeId={id} />}

      {hasInputHandle && (
        <Handle
          type="target"
          position={inputHandlePosition.position}
          style={{ position: 'absolute', ...inputHandlePosition.style }}
          className={HANDLE_CLASS}
        />
      )}

      <NodeHeader icon={icon} color={color} label={data.label} consumesCredits={consumesCredits} />

      {children && (
        <div className="mt-2 space-y-2 w-full min-w-0" onPointerDown={(e) => e.stopPropagation()}>
          {children}
        </div>
      )}

      {hasOutputHandle && (
        <Handle
          type="source"
          position={outputHandlePosition.position}
          style={{ position: 'absolute', ...outputHandlePosition.style }}
          className={HANDLE_CLASS}
        />
      )}

      {!readOnly && !dragging && (outputButtons ?? (hasOutputHandle && <OutputEdgeButton nodeId={id} nodeType={data.type} />))}
    </div>
  );
});
